import os
import site
from pathlib import Path
site.addsitedir(str(Path(os.environ.get('CEDIRNET_STEM_SOURCE',os.path.join(os.environ.get('TOOLBOX_CACHE','.'),'cedirnet-stem'))) / 'src'))

import torch
import modelargs
from stem_plugin.serving import load_runtime, load_pairs, preannotation

if __name__ == '__main__':
    import argparse
    import sys
    # modelargs consumes options after '--' and rewrites sys.argv. Parse image
    # inputs first; accept the inference-only modality on either side of it.
    input_args, model_options = sys.argv[1:], []
    if '--' in input_args:
        separator = input_args.index('--')
        input_args, model_options = input_args[:separator], input_args[separator+1:]
    mode_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mode_parser.add_argument('--modality', choices=('paired', 'BF', 'HAADF'))
    model_mode, model_options = mode_parser.parse_known_args(model_options)
    input_parser = argparse.ArgumentParser(allow_abbrev=False)
    input_parser.add_argument('images', nargs='+')
    input_parser.add_argument('--modality', choices=('paired', 'BF', 'HAADF'))
    CLI_ARGS = input_parser.parse_args(input_args)
    if CLI_ARGS.modality is None:
        CLI_ARGS.modality = model_mode.modality
    sys.argv = [sys.argv[0], '--', *model_options]

CMD_ARGS = modelargs.parse('./model.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUNTIME = load_runtime(CMD_ARGS,DEVICE)


def predict(images, interactive=False, input_mode='paired'):
    # The browser must retain low-score candidates so its slider is reversible.
    # CLI and Label Studio preannotations still use the configured cutoff.
    cutoff = 0 if interactive else CMD_ARGS['score_threshold']
    response = RUNTIME.predict(images,size=(CMD_ARGS['width'],CMD_ARGS['height']),score_threshold=cutoff)
    response['candidate_score_threshold'] = cutoff
    response['display_score_threshold'] = CMD_ARGS['score_threshold']
    response['input_mode'] = input_mode
    return response


if __name__ == '__main__':
    import json
    response = predict(load_pairs(CLI_ARGS.images, modality=CLI_ARGS.modality),
                       input_mode=CLI_ARGS.modality or 'paired')
    Path('result.json').write_text(json.dumps(response))
else:
    from flask import request
    from label_studio_ml.api import init_app
    from label_studio_ml.model import LabelStudioMLBase
    from label_studio_ml.response import ModelResponse

    class CeDiRNetSTEM(LabelStudioMLBase):
        def predict(self,tasks,context=None,**kwargs):
            # don't return interactive predictions
            if context is not None:
                return

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
            modality = request.form.get('modality')
            return predict(load_pairs(request.files.getlist('images'), modality=modality),
                           interactive=True, input_mode=modality or 'paired')
        except (ValueError,FileNotFoundError,OSError) as error:
            return {'error':str(error)},400
