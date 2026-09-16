import os
import site
from pathlib import Path
site.addsitedir(str(Path(os.environ.get('CEDIRNET_STEM_SOURCE',os.path.join(os.environ.get('TOOLBOX_CACHE','.'),'cedirnet-stem'))) / 'src'))

import torch
import modelargs
from stem_plugin.serving import load_runtime, load_pairs, preannotation

CMD_ARGS = modelargs.parse('./model.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUNTIME = load_runtime(CMD_ARGS,DEVICE)


def predict(images):
    return RUNTIME.predict(images,size=(CMD_ARGS['width'],CMD_ARGS['height']),score_threshold=CMD_ARGS['score_threshold'])


if __name__ == '__main__':
    import sys
    import json
    response = predict(load_pairs(sys.argv[1:3]))
    Path('result.json').write_text(json.dumps(response))
else:
    from flask import request
    from label_studio_ml.api import init_app
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse

    class CeDiRNetSTEM(LabelStudioMLBase):
        def predict(self,tasks,context=None,**kwargs):
            predictions = []
            for task in tasks:
                paths = task['data'].get('images')
                if not isinstance(paths,list) or len(paths)!=2:
                    raise ValueError('STEM tasks require data.images [BF, HAADF]')
                image = load_pairs([self.get_local_path(p,task_id=task.get('id')) for p in paths])[0]
                response = predict([image])
                predictions.append(preannotation(response,0,(image.shape[1],image.shape[0]),self.parsed_label_config))
            return ModelResponse(predictions=predictions)

    app = init_app(model_class=CeDiRNetSTEM)

    @app.route('/infer',methods=['POST'])
    def infer():
        try:
            return predict(load_pairs(request.files.getlist('images')))
        except (ValueError,FileNotFoundError,OSError) as error:
            return {'error':str(error)},400
