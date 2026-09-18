"""Exercise the real host components in Chromium; predictions are synthetic, not accuracy evidence."""
from pathlib import Path
import json, base64, zipfile, io, os, tempfile
import numpy as np
from playwright.sync_api import sync_playwright
from PIL import Image
OUT=Path(os.environ.get('STEM_BROWSER_ARTIFACTS', tempfile.mkdtemp(prefix='stem-browser-')))
OUT.mkdir(parents=True, exist_ok=True)
URL='http://127.0.0.1:'+os.environ.get('STEM_BROWSER_PORT','18768')
report={'fixture':'Synthetic predictions; actual host model.js, settings(), CSS, upload/change/fetch/infer lifecycle', 'checks':[]}
def check(name, value):
    assert value, name
    report['checks'].append(name)
def decode(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1]))).convert('RGB')
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path=os.environ.get('STEM_BROWSER_EXECUTABLE'),headless=True,args=['--no-sandbox','--disable-gpu'])
    page=browser.new_page(viewport={'width':1280,'height':960},accept_downloads=True)
    errors=[];console=[];network=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:requests.append({'method':r.method,'url':r.url}))
    page.on('requestfailed',lambda r:network.append({'url':r.url,'failure':r.failure}))
    page.on('response',lambda r:network.append({'url':r.url,'status':r.status}) if r.status>=400 else None)
    page.on('console',lambda m:console.append(m.text) if m.type=='error' else None)
    page.goto(URL+'/')
    def ready(count=None):
        page.wait_for_function("document.querySelector('.stem-results')?.dataset.revision && document.querySelector('.stem-results').getAttribute('aria-busy')==='false'")
        if count is not None: page.wait_for_function('(n)=>document.querySelector(".stem-results").dataset.particleCount===String(n)',arg=count)
    def control(name, value):
        page.locator('[data-control='+name+']').evaluate('(e,v)=>{e.value=String(v);e.dispatchEvent(new Event("input"))}',value)
        page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>resolve()))')
    def clicktext(text): page.get_by_role('button',name=text,exact=True).click()
    def pixels(): return page.locator('.stem-annotated').evaluate_all('els=>els.map(c=>c.toDataURL())')
    def dimensions(): return page.locator('.stem-annotated').evaluate_all('els=>els.map(c=>[c.width,c.height,c.clientWidth,c.clientHeight])')
    def download(label, filename):
        with page.expect_download() as pending: clicktext(label)
        pending.value.save_as(OUT/filename)
        return OUT/filename
    ready(2)
    check('top downloads have exact labels',page.locator('#results button').all_text_contents()==['Download detections','Download mask','Download images'])
    check('no inline per-sample mask links',page.locator('#results a').count()==0)
    check('both modalities and samples retain native dimensions and visible proportional layout',dimensions()==[[800,400,512,256],[800,400,512,256],[400,200,400,200],[400,200,400,200]])
    report['image_dimensions']=dimensions();baseline=pixels()
    for index,image in enumerate(page.locator('.stem-annotated').all()):
        image.scroll_into_view_if_needed()
        visible=Image.open(io.BytesIO(image.screenshot(path=str(OUT/f'visible-image-{index}.png')))).convert('RGB')
        check(f'image {index}: viewport overlay pixels visible',len(visible.getcolors(1000000))>3)
    check('no captions or status prose',page.locator('#results figcaption, #results [role=status]').count()==0)
    response=page.request.get(URL+'/response.json').json()
    expected=[{'name':n,'color':'rgb('+', '.join(map(str,c))+')'} for n,c in zip(response['segmentation'][0]['classes'],response['segmentation'][0]['colors'])]
    check('compact legend matches class names and colors',page.locator('.stem-class-legend > span').evaluate_all('els=>els.map(e=>({name:e.textContent,color:e.firstElementChild.style.backgroundColor}))')==expected)
    page.get_by_role('button',name='Download detections',exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(OUT/'combined.png'),full_page=True)
    clicktext('STEM visualization settings')
    check('native settings opens',page.locator('dialog').evaluate('(d)=>d.open'))
    check('no Ok or Cancel buttons',page.get_by_role('button',name='Ok',exact=True).count()+page.get_by_role('button',name='Cancel',exact=True).count()==0)
    page.screenshot(path=str(OUT/'settings.png'),full_page=True)
    # Count expensive browser operations after initial preparation. No rerender may decode/encode.
    page.evaluate('''()=>{window.costs={decode:0,encode:0};
      const blob=HTMLCanvasElement.prototype.toBlob;
      HTMLCanvasElement.prototype.toBlob=function(...args){costs.encode++;return blob.apply(this,args)};
      const src=Object.getOwnPropertyDescriptor(HTMLImageElement.prototype,'src');
      Object.defineProperty(HTMLImageElement.prototype,'src',{...src,set(v){costs.decode++;src.set.call(this,v)}});
    }''')
    before=len(requests)
    latency=page.evaluate('''async()=>{
      const input=document.querySelector('[data-control=threshold]');
      const c=document.querySelector('.stem-annotated');
      const old=c.getContext('2d').getImageData(224,200,1,1).data.slice();
      const revision=document.querySelector('.stem-results').dataset.revision;
      const start=performance.now();
      for(const v of [.95,.5,.9,0]){input.value=v;input.dispatchEvent(new Event('input'));}
      return await new Promise(resolve=>requestAnimationFrame(()=>{
        const group=document.querySelector('.stem-results');
        const pixel=c.getContext('2d').getImageData(224,200,1,1).data;
        resolve({ms:performance.now()-start,count:group.dataset.particleCount,
          draws:Number(group.dataset.revision)-Number(revision),pixelChanged:pixel.some((v,i)=>v!==old[i])});
      }));
    }''')
    report['next_frame_latency']=latency
    check('rapid inputs draw latest filters in the next animation frame',latency['count']=='4' and latency['draws']==1 and latency['pixelChanged'])
    control('minRadius',25);ready(1);check('radius uses original pixels',True)
    control('minRadius',0);ready(4)
    page.locator('[data-control=minRadius]').press('Enter')
    check('Enter in a display control cannot submit or reset host inference',len(requests)==before and page.locator('[data-control=threshold]').count()==1)
    control('threshold',.95);ready(1)
    check('score threshold accepts scores above one',True)
    page.locator('[data-class-name="Film"]').uncheck();ready(1)
    page.evaluate('()=>new Promise(resolve=>requestAnimationFrame(()=>resolve()))')
    hidden=pixels()
    check('legend reflects selection',page.locator('.stem-class-legend > span').all_text_contents()==['Carbon','Vacuum'])
    check('no image decoding or encoding on filter changes',page.evaluate('costs')=={'decode':0,'encode':0})
    check('no requests or inference on score/radius/class changes',len(requests)==before)
    report['filter_requests']=requests[before:];report['filter_costs']=page.evaluate('costs')
    page.screenshot(path=str(OUT/'filtered-settings.png'),full_page=True)
    clicktext('Close');ready(1)
    archive=zipfile.ZipFile(download('Download images','filtered.zip'))
    check('ZIP contains all four JPEG modality plots',len(archive.namelist())==4 and all(n.endswith('.jpg') for n in archive.namelist()))
    for i,name in enumerate(archive.namelist()):
        decoded=Image.open(io.BytesIO(archive.read(name))).convert('RGB');displayed=decode(hidden[i])
        delta=np.abs(np.asarray(decoded).astype(float)-np.asarray(displayed).astype(float))
        check(f'{name}: JPEG matches display within lossy tolerance',delta.mean()<3)
        check(f'{name}: native dimensions',decoded.size==displayed.size)
        w,h=decoded.size
        check(f'{name}: hidden Film restores background',max(abs(v-(40 if i%2==0 else 80)) for v in decoded.getpixel((w//2,30)))<=3)
        check(f'{name}: visible other class unchanged',max(abs(a-b) for a,b in zip(decoded.getpixel((30,30)),decode(baseline[i]).getpixel((30,30))))<=3)
        point=(639,200) if i<2 else (319,100)
        check(f'{name}: circle radius and threshold',decode(baseline[i]).getpixel(point)[1]>150 and (decoded.getpixel(point)[1]<150 if i<2 else decoded.getpixel(point)[1]>150))
    masks=zipfile.ZipFile(download('Download mask','masks.zip'))
    check('mask ZIP contains every sample',len(masks.namelist())==2)
    for i,name in enumerate(masks.namelist()):
        mask=Image.open(io.BytesIO(masks.read(name)))
        original=Image.open(io.BytesIO(base64.b64decode(response['segmentation'][i]['mask_png'].split(',')[1])))
        check(f'{name}: raw mask exact unfiltered class IDs and dimensions',mask.size==original.size and mask.tobytes()==original.tobytes() and set(np.asarray(mask).ravel())=={0,1,2})
    detections=json.loads(download('Download detections','detections.json').read_text())
    check('JSON retains every unfiltered candidate', [r['scores'] for r in detections]==response['scores'])
    clicktext('STEM visualization settings');control('threshold',0);ready(4);page.keyboard.press('Escape');ready(4)
    check('Escape closes without rollback',not page.locator('dialog').evaluate('(d)=>d.open'))
    page.reload();ready(4)
    check('live settings persist without confirmation',not page.locator('[data-class-name="Film"]').is_checked())
    clicktext('STEM visualization settings');clicktext('None');ready(4)
    check('None hides legend',not page.locator('.stem-class-legend').is_visible())
    clicktext('All');ready(4);check('All restores visibility',all(page.locator('[data-class-name]').evaluate_all('els=>els.map(e=>e.checked)')));clicktext('Close')
    for mode in ['particles','segmentation','empty']:
        page.evaluate('(mode)=>runFixture(mode)',mode);page.wait_for_timeout(300);ready()
        particle=mode!='segmentation';semantic=mode=='segmentation'
        check(mode+': one native settings control',page.locator('.settings').count()==1)
        check(mode+': visible images',page.locator('.stem-annotated').evaluate_all('els=>els.every(i=>i.clientWidth>0 && i.clientHeight>0)'))
        check(mode+': legend only for included segmentation',page.locator('.stem-class-legend').count()==int(semantic))
        check(mode+': thresholds only for included nanoparticles',page.locator('[data-control=threshold]').count()==int(particle))
        check(mode+': class controls only for included segmentation',page.locator('[data-class-name]').count()==(3 if semantic else 0))
        check(mode+': detections download task-aware',page.get_by_role('button',name='Download detections',exact=True).count()==int(particle))
        check(mode+': mask download task-aware',page.get_by_role('button',name='Download mask',exact=True).count()==int(semantic))
        check(mode+': images download present',page.get_by_role('button',name='Download images',exact=True).count()==1)
        if mode=='empty':
            check('enabled particle task with no detections retains usable controls',not page.locator('[data-control=threshold]').is_disabled())
            empty=json.loads(download('Download detections','empty.json').read_text())
            check('empty particle JSON remains valid',all(r['scores']==[] for r in empty))
    page.evaluate('document.querySelector("form").reset()');check('reset disposes settings',page.locator('.settings').count()==0)
    page.evaluate("localStorage.setItem('toolbox-ui-stem-inference','broken');runFixture()")
    page.wait_for_timeout(300);ready(2);check('corrupt storage falls back',True)
    if os.environ.get('STEM_REAL_PNG_DIR'):
        real=Path(os.environ['STEM_REAL_PNG_DIR']);files=[real/f'PtCo_IL_a-0016_{m}.png' for m in ['BF','HAADF']]
        page.locator('#infer input[type=file]').set_input_files(files)
        page.wait_for_function('document.querySelectorAll(".stem-annotated").length===2');ready(1)
        report['real_sample_dimensions']=dimensions()
        check('real BF/HAADF decode at native dimensions',[x[:2] for x in dimensions()]==[list(Image.open(f).size) for f in files])
        clicktext('STEM visualization settings');clicktext('None');control('threshold',1);clicktext('Close');ready(0)
        real_pixels=pixels()
        for index,image in enumerate(page.locator('.stem-annotated').all()):
            image.scroll_into_view_if_needed();image.screenshot(path=str(OUT/f'real-{index}-visible.png'))
            check(f'real modality {index}: overlays off restores exact source pixels',decode(real_pixels[index]).tobytes()==Image.open(files[index]).convert('RGB').tobytes())
        page.locator('.stem-annotated').first.scroll_into_view_if_needed();page.screenshot(path=str(OUT/'real-sample.png'))
    if errors or console or network: print(json.dumps({'errors':errors,'console':console,'network':network},indent=2))
    check('no failed requests',network==[]);check('no console errors',console==[]);check('no JS errors',errors==[])
    report.update(errors=errors,network_errors=network,console_errors=console,zip_members=archive.namelist())
    browser.close()
report['check_count']=len(report['checks'])
(OUT/'browser-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
