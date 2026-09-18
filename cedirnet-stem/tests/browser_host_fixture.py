"""Local browser contract harness: actual host JS/CSS, actual plugin UI, explicit synthetic inference fixtures."""
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import sys, json, io, os
import numpy as np
from PIL import Image
PLUGIN = Path(__file__).resolve().parents[1]
HOST = Path(os.environ['TOOLBOX_HOST']) / 'apps/nexus'
sys.path.insert(0, str(PLUGIN))
from stem_plugin.semantic_results import encode_mask
response = dict(centers=[[[.25,.5],[.75,.5]], [[.25,.5],[.75,.5]]], scores=[[.2,.9],[.4,1.2]], radii=[[24,40],[12,20]], candidate_score_threshold=0,display_score_threshold=.5, tasks={'nanoparticles':True,'segmentation':True})
response['segmentation'] = []
for w,h in [(800,400),(400,200)]:
    mask = np.tile((np.arange(w)*3//w).astype(np.uint8), (h,1))
    mask[-8:, -8:] = 255  # Ignore is not a class or background.
    response['segmentation'].append(encode_mask(mask, ['Carbon','Film','Vacuum']))
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        result=json.loads(json.dumps(response))
        mode=self.path.split('mode=')[-1]
        result['tasks'] = {'nanoparticles': mode != 'segmentation', 'segmentation': mode not in ('particles', 'empty')}
        if mode in ('particles','empty'): result['segmentation']=[None,None]
        if mode in ('segmentation','empty'):
            for key in ('centers','scores','radii'): result[key]=[[],[]]
        data=json.dumps(result).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == '/':
            html = '''<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><link rel="stylesheet" href="/static/style.css"><title>STEM inference controls verification</title><header class="toolbar" style="z-index:2"><div class="toolbar-left">STEM · browser contract harness (synthetic fixtures)</div><div class="toolbar-right"></div></header><main style="padding:4rem 1rem 1rem;overflow:auto"><form style="position:relative">PLUGIN</form></main><script src="/static/model.js"></script><script>
window.loading=()=>({remove(){}}); window.showToast=(level,e)=>console.error(e); window.errors=[]; addEventListener('error',e=>errors.push(e.message));
window.runFixture = async function(mode='combined') {
 const data=new DataTransfer();
 for(const file of ['0-BF.png','0-HAADF.png','1-BF.png','1-HAADF.png']) data.items.add(new File([await (await fetch('/'+file)).blob()],file));
 document.querySelector('form').action='/infer?mode='+mode;
 const input=document.getElementById('infer').shadowRoot.querySelector('input[type=file]');
 input.files=data.files; input.dispatchEvent(new Event('change'));
};
runFixture();
</script>'''
            ui=PLUGIN.joinpath('ui.html').read_text()
            # Use the real model-page containment (including fieldset overflow), not a free-standing form.
            template=(HOST/'templates/model.html').read_text()
            worker=template.split('<form class="inference-tab"')[1].split('</form>')[0]
            worker='<form class="inference-tab"'+worker+'</form>'
            worker=worker.replace('{{ alias }}','fixture').replace('{{ form | safe }}',ui)
            start=html.index('<header'); end=html.index('</main>')+len('</main>')
            html=html[:start]+'<nav>Toolbox</nav><main><div class="model-overview wrapper"><div class="inference"><details name="workers" open><summary>fixture</summary>'+worker+'</details></div></div></main>'+html[end:]
            data=html.encode(); mime='text/html'
        elif self.path == '/response.json': data=json.dumps(response).encode(); mime='application/json'
        elif self.path.startswith('/static/'):
            p=HOST/self.path.lstrip('/')
            if self.path == '/static/properties.css': p=HOST.parents[1]/'branding/properties.css'
            data=p.read_bytes(); mime='text/css' if p.suffix=='.css' else 'image/svg+xml' if p.suffix=='.svg' else 'text/javascript'
        elif self.path.endswith('.png'):
            w,h=(800,400) if self.path.startswith('/0') else (400,200)
            color=40 if 'BF' in self.path else 80
            out=io.BytesIO(); Image.new('RGB',(w,h),(color,)*3).save(out,format='PNG');data=out.getvalue();mime='image/png'
        else: self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.end_headers();self.wfile.write(data)
if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('STEM_BROWSER_PORT', '18768'))), Handler).serve_forever()
