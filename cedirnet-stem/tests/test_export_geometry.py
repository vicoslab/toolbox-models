"""Schema regressions: LS combined controls and separate geometry + Labels."""
import copy
import json
import numpy as np
import pytest
from PIL import Image
from ls_adapter import export


def region(kind, label, **geometry):
    return dict(id=kind+label, from_name='semantic', to_name='image', type=kind+'labels',
                original_width=64, original_height=64, image_rotation=0, item_index=0,
                value={kind+'labels': [label], **geometry})


def shapes(kind, label):
    if kind == 'rectangle': return region(kind,label,x=10,y=10,width=20,height=20,rotation=0)
    if kind == 'ellipse': return region(kind,label,x=20,y=20,radiusX=10,radiusY=10,rotation=30)
    return region(kind,label,points=[[10,10],[30,10],[30,30],[10,30]])


@pytest.mark.parametrize('kind',['polygon','rectangle','ellipse'])
@pytest.mark.parametrize('label,cls',[('Carbon',0),('Film',1),('Vacuum',2),('Ignore',255)])
def test_all_geometry_classes(tmp_path,kind,label,cls):
    out=export([[shapes(kind,label)]],tmp_path,['BF','HAADF'],False)
    assert 'points' not in out
    mask=np.array(Image.open(tmp_path/out['semantic_mask']))
    assert mask[12,12] == cls
    assert mask[63,63] == 255


def test_separate_labels_particle_and_frame_dedup(tmp_path):
    geometry=region('ellipse','nanoparticle',x=25,y=25,radiusX=10,radiusY=10)
    geometry['type']='ellipse'; geometry['from_name']='geometry'; del geometry['value']['ellipselabels']
    label=dict(id=geometry['id'],from_name='labels',to_name='image',type='labels',item_index=0,value={'labels':['nanoparticle']})
    pair=copy.deepcopy([geometry,label])
    for t in pair: t['item_index']=1
    out=export([[label,geometry,*pair]],tmp_path,['BF','HAADF'],False)
    assert out['points']==[[16,16,6.4]]


@pytest.mark.parametrize('kind',['polygon','rectangle','ellipse'])
def test_separate_semantic_labels(tmp_path,kind):
    geometry=shapes(kind,'Film'); geometry['type']=kind
    del geometry['value'][kind+'labels']
    label=dict(id=geometry['id'],from_name='semantic',to_name='image',type='labels',item_index=0,value={'labels':['Film']})
    geometry['from_name']='geometry'
    out=export([[label,geometry]],tmp_path,['BF','HAADF'],False)
    assert np.array(Image.open(tmp_path/out['semantic_mask']))[12,12]==1


def test_unlabelled_geometry_rejected(tmp_path):
    tag=shapes('ellipse','Carbon');tag['type']='ellipse';tag['value'].pop('ellipselabels')
    with pytest.raises(ValueError,match='label'):
        export([[tag]],tmp_path,['BF','HAADF'],False)


def test_real_magicwand_paired_brush(tmp_path):
    from pathlib import Path
    tags=json.loads((Path(__file__).parent/'fixtures/ls-1.23-magicwand.json').read_text())
    wand=next(t for t in tags if t['type']=='magicwand')
    paired=[t for t in tags if t['id']==wand['id']]
    assert {t['type'] for t in paired} == {'magicwand','brushlabels'}
    out=export([paired],tmp_path,['BF','HAADF'],False)
    assert np.all(np.array(Image.open(tmp_path/out['semantic_mask'])) == 0)
    with pytest.raises(ValueError,match='label'):
        export([[wand]],tmp_path,['BF','HAADF'],False)


@pytest.mark.parametrize('class_index',range(4))
def test_browser_magicwand_all_classes(tmp_path,class_index):
    from pathlib import Path
    fixture=Path(__file__).parent/'fixtures/ls-1.23-magicwand-classes.json'
    tags=json.loads(fixture.read_text())[class_index]['results']
    out=export([tags],tmp_path,['BF','HAADF'],False)
    assert np.all(np.array(Image.open(tmp_path/out['semantic_mask'])) == [0,1,2,255][class_index])


@pytest.mark.parametrize('kind',['polygon','rectangle','ellipse'])
def test_browser_separate_geometry(tmp_path,kind):
    from pathlib import Path
    tags=json.loads((Path(__file__).parent/'fixtures/ls-1.23-separate.json').read_text())
    out=export([[t for t in tags if t['id']==kind]],tmp_path,['BF','HAADF'],False)
    assert np.array(Image.open(tmp_path/out['semantic_mask']))[12,12]==1


def test_union_conflict_and_rotation(tmp_path):
    first=region('rectangle','Carbon',x=10,y=10,width=20,height=10,rotation=90)
    second=region('rectangle','Carbon',x=60,y=60,width=20,height=10,rotation=0)
    film=region('rectangle','Film',x=60,y=60,width=10,height=10,rotation=0)
    a=export([[first,second,film]],tmp_path,['BF','HAADF'],False)
    b=export([[film,second,first]],tmp_path,['BF','HAADF'],False)
    assert a==b
    m=np.array(Image.open(tmp_path/a['semantic_mask']))
    assert m[12,3]==0 and m[3,12]==255
    assert m[40,40]==255 and m[40,48]==0


@pytest.mark.parametrize('case',['unknown','nonfinite','open','unmatched','frame','mismatch'])
def test_reject_dropped_or_ambiguous_regions(tmp_path,case):
    tag=shapes('polygon','Carbon')
    if case=='unknown': tag['value']['polygonlabels']=['Unknown']
    if case=='nonfinite': tag['value']['points'][0][0]=float('nan')
    if case=='open': tag['value']['closed']=False
    if case=='frame': tag['item_index']=2
    tags=[tag]
    if case in ('unmatched','mismatch'):
        tag['type']='polygon';del tag['value']['polygonlabels']
        label=dict(id=tag['id'],from_name='semantic',to_name='image',type='labels',item_index=0,value={'labels':['Carbon']})
        if case=='unmatched': label['id']='other'
        else: label['original_width']=100
        tags.append(label)
    with pytest.raises(ValueError): export([tags],tmp_path,['BF','HAADF'],False)


def test_semantic_aliases_follow_class_order(tmp_path,monkeypatch):
    import ls_adapter
    monkeypatch.setattr(ls_adapter,'semantic_aliases',lambda:{'c':'Carbon','f':'Film','v':'Vacuum','i':'Ignore'})
    out=export([[shapes('rectangle','f')]],tmp_path,['BF','HAADF'],False)
    assert out['semantic_classes']==['Carbon','Film','Vacuum']
    assert np.array(Image.open(tmp_path/out['semantic_mask']))[12,12]==1


def test_semantic_preannotation_alias_roundtrip(tmp_path,monkeypatch):
    import ls_adapter
    from label_studio_sdk._extensions.label_studio_tools.core.label_config import parse_config
    from stem_plugin.serving import preannotation
    from stem_plugin.semantic_results import encode_mask
    config=parse_config('<View><Image name="image" value="$image"/><BrushLabels name="semantic" toName="image"><Label value="Carbon" alias="c"/><Label value="Film" alias="f"/><Label value="Vacuum" alias="v"/></BrushLabels></View>')
    mask=np.tile(np.array([0,1,2],dtype=np.uint8),(3,1))
    response=dict(tasks={'nanoparticles':False},scores=[[]],segmentation=[encode_mask(mask,['Carbon','Film','Vacuum'])])
    annotation=preannotation(response,0,(3,3),config)
    assert [r['value']['brushlabels'] for r in annotation['result']]==[['c'],['f'],['v']]
    monkeypatch.setattr(ls_adapter,'semantic_aliases',lambda:{'c':'Carbon','f':'Film','v':'Vacuum'})
    out=export([annotation['result']],tmp_path,['BF','HAADF'],False)
    assert np.array_equal(mask,np.array(Image.open(tmp_path/out['semantic_mask'])))


def test_distinct_coincident_particles_are_not_dropped(tmp_path):
    from test_ellipse import ellipse
    first=ellipse();second=copy.deepcopy(first);second['id']='other'
    assert len(export([[first,second]],tmp_path,['BF','HAADF'],False)['points'])==2


def test_unsupported_spatial_result_is_not_dropped(tmp_path):
    tag=region('keypoint','Carbon',x=20,y=20)
    tag['from_name']='other-control'
    with pytest.raises(ValueError,match='unsupported'):
        export([[tag]],tmp_path,['BF','HAADF'],False)


@pytest.mark.parametrize('kind',['polygon','rectangle','ellipse','Carbon','Film','Vacuum','Ignore'])
def test_real_tool_host_export_dataset(tmp_path,monkeypatch,kind):
    from pathlib import Path
    from test_export import host_export
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.stem_tasks import TaskConfig
    fixtures=Path(__file__).parent/'fixtures'
    if kind in ('polygon','rectangle','ellipse'):
        tags=[t for t in json.loads((fixtures/'ls-1.23-separate.json').read_text()) if t['id']==kind]
        expected=1
    else:
        rows=json.loads((fixtures/'ls-1.23-magicwand-classes.json').read_text())
        tags=next(row['results'] for row in rows if row['label']==kind)
        expected={'Carbon':0,'Film':1,'Vacuum':2,'Ignore':255}[kind]
    particle=next(t for t in json.loads((fixtures/'ls-1.23-current.json').read_text()) if t['type']=='ellipselabels')
    manifest=host_export(tmp_path,monkeypatch,[*tags,particle])
    for split in ('train','val','test','data'):
        sample=ToolboxDataset(manifest,TaskConfig(True,True),split=split,size=(64,64))[0]
        assert sample['shape_coef'][0,16,16]==8
        assert sample['semantic_segmentation'][12,12]==expected


def test_missing_points_diagnostic(tmp_path):
    from stem_plugin.toolbox_dataset import ToolboxDataset
    from stem_plugin.stem_tasks import TaskConfig
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'version':4,'train':[{'images':['BF.png','HAADF.png']}]}))
    with pytest.warns(UserWarning, match=r'train.*kept 0 of 1, skipped 1'):
        with pytest.raises(ValueError, match=r'No usable samples.*points.*re-export'):
            ToolboxDataset(manifest,TaskConfig(True,False))
