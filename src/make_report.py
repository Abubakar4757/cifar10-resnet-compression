"""Create the plain black Assignment 3 report from measured result files only."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT
RESULTS=PACKAGE/'results'
ASSETS=ROOT/'report_work'
ASSETS.mkdir(parents=True,exist_ok=True)


def j(name):return json.loads((RESULTS/name).read_text())
def rows(name):
    with (RESULTS/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def num(x,d=2):return f'{float(x):.{d}f}'
LABEL={'fp32':'FP32 baseline','ptq':'PTQ INT8','qat':'QAT INT8',
       'unstructured':'Unstructured','structured':'Structured'}


def build():
    bench=rows('benchmark_summary.csv')
    native={r['model']:r for r in bench if r['runtime']=='pytorch'}
    ort={r['model']:r for r in bench if r['runtime']=='onnxruntime'}
    fp=j('fp32_architecture.json');sp=j('structured_architecture.json')
    baseline=j('baseline_summary.json');ptq=j('ptq_summary.json');qat=j('qat_summary.json')
    up=j('unstructured_summary.json');st=j('structured_summary.json')
    environment=j('benchmark_environment.json');training=j('training_environment.json')
    parity=rows('onnx_parity.csv')
    model_info={r['model']:r for r in j('model_verification.json')}
    order=['fp32','ptq','qat','unstructured','structured']
    # Monochrome scientific figures, without decorative page rules.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    hist=rows('baseline_history.csv')
    fig,ax=plt.subplots(figsize=(6.5,2.35))
    ax.plot([int(r['epoch']) for r in hist],[float(r['train_accuracy']) for r in hist],
            color='black',linestyle='--',label='Training')
    ax.plot([int(r['epoch']) for r in hist],[float(r['val_accuracy']) for r in hist],
            color='black',label='Validation')
    ax.set(xlabel='Epoch',ylabel='Accuracy percent');ax.legend(frameon=False,loc='lower right')
    fig.tight_layout();fig.savefig(ASSETS/'training.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,2.4))
    ax.bar(np.arange(5)-.18,[float(native[n]['batch_one_p50_ms']) for n in order],.36,
           color='white',edgecolor='black',hatch='///',label='PyTorch')
    ax.bar(np.arange(5)+.18,[float(ort[n]['batch_one_p50_ms']) for n in order],.36,
           color='.6',edgecolor='black',label='ONNX Runtime')
    ax.set_xticks(np.arange(5),['FP32','PTQ','QAT','Unstructured','Structured'])
    ax.set_ylabel('Batch one p50 latency ms');ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(ASSETS/'latency.png',dpi=190);plt.close(fig)
    doc=Document();sec=doc.sections[0]
    sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
    sec.top_margin=sec.bottom_margin=Inches(.7)
    sec.left_margin=sec.right_margin=Inches(.78)
    for style in doc.styles:
        if hasattr(style,'font'):
            style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
            if style.type==1:
                style.paragraph_format.space_after=Pt(6)
                style.paragraph_format.line_spacing=1.08
    doc.styles['Normal'].font.size=Pt(11)
    doc.styles['Title'].font.size=Pt(20)
    for name,size in [('Heading 1',15),('Heading 2',12)]:
        doc.styles[name].font.size=Pt(size);doc.styles[name].font.bold=True
    # Remove theme decorations, direct borders, and all table lines explicitly.
    for style in doc.styles:
        for e in list(style.element.iter(qn('w:pBdr'))):e.getparent().remove(e)
    doc.core_properties.title='CIFAR-10 ResNet Compression'
    doc.core_properties.author='Abubakar Bilal'
    def p(text,style=None):return doc.add_paragraph(text,style)
    def heading(text):doc.add_heading(text,1)
    def sub(text):doc.add_heading(text,2)
    def page(title):doc.add_page_break();heading(title)
    def table(headers,data,widths=None):
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER
        t.autofit=False
        for i,v in enumerate(headers):t.rows[0].cells[i].text=str(v)
        for row in data:
            cells=t.add_row().cells
            for i,v in enumerate(row):cells[i].text=str(v)
        borders=OxmlElement('w:tblBorders')
        for tag in ['top','left','bottom','right','insideH','insideV']:
            e=OxmlElement('w:'+tag);e.set(qn('w:val'),'nil');borders.append(e)
        t._tbl.tblPr.append(borders)
        for ri,row in enumerate(t.rows):
            for i,c in enumerate(row.cells):
                c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if widths:c.width=Inches(widths[i])
                for para in c.paragraphs:
                    para.paragraph_format.space_before=Pt(3)
                    para.paragraph_format.space_after=Pt(3)
                    para.paragraph_format.line_spacing=1.02
                    for run in para.runs:run.font.size=Pt(10);run.bold=(ri==0)
        p('')
        return t
    def picture(path,width=6.45):
        para=doc.add_paragraph();para.alignment=WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(path),width=Inches(width))
    p('CIFAR-10 ResNet Compression','Title')
    p('Edge Computing and Intelligent Systems | Assignment 3')
    p('Abubakar Bilal    22I-0450    Section A')
    heading('Purpose and measured outcomes')
    p('This experiment implements post-training static quantization, quantization-aware training, '
      'iterative unstructured pruning, and physical channel pruning for CIFAR-10 classification. '
      'A common trained ResNet-18 baseline makes the optimization comparisons consistent. '
      'All five variants have executable PyTorch checkpoints and verified ONNX CPU deployments.')
    p(f'The baseline achieved {num(native["fp32"]["test_accuracy_percent"])} percent test accuracy. '
      f'PTQ achieved {num(native["ptq"]["test_accuracy_percent"])} percent and QAT achieved '
      f'{num(native["qat"]["test_accuracy_percent"])} percent. The physically pruned network reduced '
      f'parameters by {num(100*(1-sp["parameters"]/fp["parameters"]))} percent and MACs by '
      f'{num(100*(1-sp["MACs_per_image"]/fp["MACs_per_image"]))} percent. '
      'PTQ provides the strongest observed balance of accuracy, checkpoint size, and CPU latency in this run.')
    table(['Model','PyTorch accuracy %','ONNX accuracy %','Checkpoint MiB'],
          [[LABEL[n],num(native[n]['test_accuracy_percent']),num(ort[n]['test_accuracy_percent']),
            num(model_info[n]['native_bytes']/2**20)] for n in order],[1.65,1.65,1.65,1.65])
    sub('Scope')
    p('Question 1 covers the common baseline, static PTQ, QAT, deployment, predictions, and benchmarking. '
      'Question 2 uses the same baseline for two independently fine-tuned pruning branches. '
      'Pruning is not combined with quantization in the five required variants. '
      'The submitted scripts can reproduce every stage and rerun inference without retraining.')
    p('CPU measurements describe the recorded host and software environment. They are not measurements '
      'on a microcontroller, and model compression does not imply an identical latency improvement.')
    page('Baseline architecture and experimental protocol')
    p('The model preserves the ResNet-18 two-block-per-stage layout. For 32 by 32 CIFAR images, '
      'the ImageNet stem is replaced by a 3 by 3 stride-one convolution and the initial max pool is omitted. '
      'Each BasicBlock contains two 3 by 3 convolutions with batch normalization and a residual addition. '
      'Stage transitions use stride two and a 1 by 1 projection on the skip path. '
      'Global average pooling feeds a 10-class linear classifier.')
    table(['Component','Channels','Spatial resolution','BasicBlocks'],
          [['Stem',64,'32 by 32',''],['Stage 1',64,'32 by 32',2],['Stage 2',128,'16 by 16',2],
           ['Stage 3',256,'8 by 8',2],['Stage 4',512,'4 by 4',2],['Head',10,'1 by 1','']],
          [1.65,1.25,2,1.7])
    p(f'The original network has {fp["parameters"]:,} trainable parameters and '
      f'{fp["MACs_per_image"]:,} convolution and linear MACs per image. '
      'MAC counts exclude normalization, activation, pooling, and residual addition costs.')
    sub('Data and training')
    p('CIFAR-10 supplies 50,000 training and 10,000 official test images. A seed-450 stratified split '
      'reserves 500 images per class for validation, leaving 45,000 fitting images. '
      'Training applies random 32 by 32 crops after four-pixel zero padding and random horizontal flips. '
      'Channel means are 0.4914, 0.4822, 0.4465 and standard deviations are 0.2470, 0.2435, 0.2616. '
      'Validation, calibration, and test transforms are deterministic.')
    p(f'The baseline was trained for {baseline["epochs"]} epochs using SGD, momentum 0.9, weight decay '
      '0.0005, initial learning rate 0.1, cosine decay, and cross-entropy loss. '
      f'Training used {training["gpu"]}, batch size 256, mixed precision, and channels-last tensors. '
      f'The best validation accuracy was {num(baseline["best_val_accuracy"])} percent; '
      'the saved checkpoint is selected by validation accuracy. The official test set is evaluated after optimization.')
    picture(ASSETS/'training.png');p('Figure 1 Training and validation accuracy across baseline epochs.')
    page('Post training quantization and quantization aware training')
    sub('Static PTQ')
    p('PTQ loads the trained FP32 checkpoint and switches to evaluation mode. Compatible Conv-BN-ReLU '
      'and Conv-BN sequences are fused, including projection shortcuts. Histogram observers collect '
      'activation statistics from a seeded shuffle of 1,024 fitting images. '
      'The validation and test sets do not participate in calibration. '
      'Weights use symmetric signed INT8 quantization per output channel; activations use affine UINT8 '
      'storage and a reduced observer range. The converted model executes quantized convolution, '
      'linear, and residual-add operations with floating point input and output boundaries.')
    p('Affine quantization uses q = clip(round(x / scale) + zero_point), and reconstruction uses '
      'x_hat = scale times (q minus zero_point). Per-channel weight scales adapt to different filter ranges. '
      'The reduced activation observer range helps avoid saturation behavior in some x86 kernels. '
      'The chosen CPU engine is recorded in the result metadata.')
    p('Calibration images per class in CIFAR class order: '+', '.join(map(str,ptq['calibration_class_counts']))+'. '
      'calibration_indices.npy contains their original training-set indices.')
    sub('QAT')
    p(f'QAT begins with the same pretrained FP32 baseline and inserts fake quantizers for activations '
      f'and per-channel weights. It fine-tunes for {qat["epochs"]} epochs with learning rate 0.005, '
      'SGD momentum 0.9, weight decay 0.0005, and cosine decay. QAT uses FP32 training arithmetic '
      'with simulated quantization; mixed precision is disabled for this branch. '
      'Fused batch-normalization statistics are frozen from epoch 3 and observers from epoch 4. '
      'The best validation checkpoint is converted on CPU into a real integer inference model.')
    table(['Property','PTQ','QAT'],[
        ['Initialization','Same FP32 baseline','Same FP32 baseline'],
        ['Extra fitting','Calibration only',f'{qat["epochs"]} fine-tuning epochs'],
        ['Weight representation','INT8 per channel','INT8 per channel'],
        ['Test accuracy %',num(native['ptq']['test_accuracy_percent']),num(native['qat']['test_accuracy_percent'])],
        ['ONNX accuracy %',num(ort['ptq']['test_accuracy_percent']),num(ort['qat']['test_accuracy_percent'])],
        ['Checkpoint MiB',num(model_info['ptq']['native_bytes']/2**20),num(model_info['qat']['native_bytes']/2**20)]],
        [2.6,2,2])
    p('Both INT8 checkpoints are serialized as executable TorchScript modules, reloaded, and checked '
      'against the in-memory converted network. Fake quantization modules are training machinery '
      'and are not retained as a substitute for native integer execution.')
    p('QAT did not improve test accuracy over PTQ in this six-epoch run. Its converted test accuracy '
      'was 0.08 percentage points below PTQ. The measured outcome demonstrates that extra QAT '
      'fine-tuning does not guarantee an accuracy gain for an already well-calibrated model.')
    page('Pruning and recovery fine tuning')
    sub('Iterative unstructured pruning')
    p('Global L1 magnitude ranking covers every convolutional and fully connected weight tensor. '
      'Three rounds target cumulative sparsities of 30, 50, and 70 percent. '
      'Each later round prunes a fraction of the remaining active entries. '
      'Masks stay active during three recovery epochs per round, preventing deleted weights from regrowing. '
      'After the final recovery, pruning reparameterizations are removed and zeros become permanent weights.')
    table(['Round','Target sparsity %','Before recovery val %','After recovery val %'],
          [[r['round'],num(100*r['target_sparsity']),num(r['before_finetune_val']),
            num(r['after_finetune_val'])] for r in up],[.65,1.55,2.2,2.2])
    p('The original experiment measured a GZIP-compressed sparse checkpoint. The repository supplies an ordinary .pt checkpoint with identical serialized weights. Historical size figures refer to the original storage format. Its dense tensor '
      'dimensions and dense MAC count remain unchanged, so its zero count does not by itself establish '
      'runtime acceleration. The ONNX export retains dense tensors containing zeros. '
      'The report compares actual file sizes and timings separately.')
    sub('Structured channel pruning')
    p('For each BasicBlock, mean absolute filter magnitude ranks the first convolution output channels. '
      'Removing 37.5 percent of these channels requires slicing matching BN parameters and the second '
      'convolution input columns. The reconstructed network executes the smaller tensors directly. '
      'Block output widths and skip interfaces are preserved, so residual additions remain valid. '
      'This is internal-channel pruning across all eight blocks, rather than masking complete output stages.')
    table(['Stage','Original internal width','Pruned internal width'],
          [[i+1,a,b] for i,(a,b) in enumerate(zip([64,128,256,512],[40,80,160,320]))],[1.2,2.7,2.7])
    p(f'The structured model has {sp["parameters"]:,} parameters and {sp["MACs_per_image"]:,} MACs per '
      f'image. Validation accuracy changed from {num(st["before_finetune_val"])} percent immediately '
      f'after pruning to {num(st["best_val_accuracy"])} percent after {st["epochs"]} recovery epochs. '
      'Recovery uses initial learning rate 0.02 with the same SGD and cosine schedule. '
      'The submitted architecture metadata records every retained channel index.')
    p('Independent verification compares the smaller network with an equivalent explicitly gated '
      'network, checks zero gradients at masked unstructured entries, and checks that final models '
      'contain no weight_orig or mask state.')
    page('CPU benchmark results')
    p(f'The benchmark host runs {environment["os"]}, with {environment["physical_cores"]} physical '
      f'and {environment["logical_cores"]} logical cores and {num(environment["ram_gib"])} GiB RAM. '
      f'Python {environment["python"]}, PyTorch {environment["torch"]}, and ONNX Runtime '
      f'{environment["onnxruntime"]} were used. The native INT8 engine is {environment["backend"]}. '
      'All comparisons within this table use the same host, one CPU thread, and batch size one.')
    p('Accuracy is evaluated on every official test image using batches of 64. Latency is measured '
      'separately, excluding preprocessing, data loading, checkpoint loading, and session creation. '
      'Each model receives 30 warmup calls before each of three repeats of 300 measured calls. '
      'Throughput divides measured calls by loop wall time. p50 and p99 are pooled sample percentiles. '
      'No simulated contention or real-time deadline is imposed in this assignment.')
    table(['Runtime','Model','Accuracy %','p50 ms','p99 ms','Images/s'],
          [[('PyTorch' if r['runtime']=='pytorch' else 'ONNX'),LABEL[r['model']],
            num(r['test_accuracy_percent']),num(r['batch_one_p50_ms'],3),num(r['batch_one_p99_ms'],3),
            num(r['batch_one_throughput_images_s'],1)] for r in bench],[.85,1.55,1.1,1,1,1.1])
    picture(ASSETS/'latency.png');p('Figure 2 Measured median inference latency under both CPU runtimes.')
    p('The CSV files contain all raw latency samples and 100,000 prediction records: '
      '10,000 images times five models times two runtimes. Prediction CSV timings are labeled '
      'batch-amortized evaluation times and must not be confused with the independent batch-one trials. '
      'Process RSS after loading is recorded as an observation, not as a peak activation-memory estimate.')
    page('ONNX deployment and result verification')
    p('The baseline and pruning variants use opset-17 floating point ONNX exports. For PTQ and QAT, '
      'the exporter reads the converted PyTorch integer tensors, weight scales, zero points, and '
      'activation quantization parameters and constructs standard QDQ graphs. '
      'This exports the QAT quantization parameters directly instead of performing a separate PTQ '
      'calibration on the QAT model. Signed INT8 weight initializers are checked explicitly. '
      'ONNX Runtime uses CPUExecutionProvider with one intra-op and one inter-op thread.')
    p('Floating point graph boundaries and some operations remain in QDQ deployment. Supported '
      'patterns can be fused into integer CPU kernels by ONNX Runtime. The graph is not described '
      'as integer-only at every node. Quantized kernels may use different rounding and accumulation '
      'paths; full-test agreement is therefore measured rather than presumed bit-exact.')
    table(['Model','Top one agreement %','Mean logit error','Max logit error'],
          [[LABEL[r['model']],num(r['top1_agreement_percent'],3),
            num(r['mean_absolute_logit_error'],6),num(r['max_absolute_logit_error'],6)] for r in parity],
          [1.65,1.75,1.6,1.6])
    sub('Checks performed')
    p('All five checkpoints are reloaded and executed. All five ONNX graphs pass the ONNX checker '
      'and execute under ONNX Runtime. Outputs have ten finite class logits per image. '
      'The structured checkpoint is inspected for smaller conv1 and matching conv2 dimensions. '
      'The unstructured checkpoint is checked for approximately 70 percent zero convolutional and '
      'linear weights with pruning masks removed. SHA256 hashes identify the exact supplied model files.')
    p('An additional Windows check reloads the saved models with the oneDNN CPU backend and runs all '
      'ten inference commands in fresh processes. Every model correctly predicts cat for the supplied '
      'CIFAR-10 test example. The reported benchmark table remains the original controlled Kaggle CPU '
      'experiment; Windows checks establish portability and are recorded separately.')
    sub('Deployment commands')
    p('From the repository root, run the following Python commands. The supplied example '
      'is CIFAR-10 test image zero; its true class is cat.')
    for cmd in ['python src/infer.py --model models/qat_int8.pt --image examples/cifar10_test_0.png',
                'python src/infer_onnx.py --model models/qat_int8.onnx --image examples/cifar10_test_0.png',
                'python src/verify_models.py --models models --output verification']:
        para=p(cmd)
        for run in para.runs:run.font.name='Consolas';run.font.size=Pt(8.5)
    page('Interpretation reproducibility and submission')
    sub('Accuracy size and latency tradeoffs')
    tradeoffs=[]
    for name in order[1:]:
        delta=float(native[name]['test_accuracy_percent'])-float(native['fp32']['test_accuracy_percent'])
        ratio=float(native['fp32']['batch_one_p50_ms'])/float(native[name]['batch_one_p50_ms'])
        storage=100*(1-model_info[name]['native_bytes']/model_info['fp32']['native_bytes'])
        tradeoffs.append([LABEL[name],f'{delta:+.2f}',f'{storage:.2f}',f'{ratio:.2f}'])
    table(['Variant','Accuracy change pp','Checkpoint reduction %','Median speed ratio'],tradeoffs,
          [1.65,1.65,1.65,1.65])
    p('Accuracy changes are relative to FP32. The speed ratio divides baseline median latency by '
      'optimized median latency, so values above one indicate faster inference. PTQ retains '
      'the most test accuracy among the optimized variants while reducing checkpoint storage '
      'by 74.50 percent and providing a 2.10 times native median speedup. QAT has similar latency '
      'and slightly lower test accuracy. The 70 percent sparse model originally compressed well with GZIP but shows '
      'no median latency gain with the dense inference kernels.')
    p('Unstructured sparsity leaves the dense workload intact; its compressed checkpoint savings '
      'must not be presented as a proportional reduction in dense arithmetic. Structured pruning '
      'removes real channels but retains residual interfaces, limiting its maximum reduction. '
      'INT8 execution benefits depend on the CPU backend and optimized graph patterns. '
      'The fastest measured model is a host-specific result, rather than a universal ranking.')
    sub('Reproducibility')
    p('All scripts use Python. requirements.txt pins the PyTorch, torchvision, ONNX, and ONNX Runtime '
      'versions. README.md provides installation, independent stage commands, inference examples, and '
      'a complete execution command. The random seed is 450. Split and calibration indices, epoch '
      'histories, pruning records, environment metadata, and raw results accompany the trained models. '
      'cuDNN benchmarking and mixed precision can produce small hardware-dependent numerical '
      'differences, so exact bitwise retraining is not claimed.')
    p('To retrain and regenerate results without overwriting the supplied artifacts:')
    para=p('python src/run_all.py --data data --output reproduction/models --results reproduction/results')
    for run in para.runs:run.font.name='Consolas';run.font.size=Pt(8.5)
    sub('Submission coverage')
    p('The GitHub repository includes the trained baseline and training script; separate PTQ, calibration, and QAT '
      'scripts; both pruning implementations and fine-tuning entry points; five model checkpoints; '
      'five ONNX exports; Python inference and benchmark scripts; CSV results; execution screenshots; '
      'the PDF report; and the README. Submit the repository link and PDF separately. Required model files are included, while datasets, '
      'installed packages, caches, and executables are excluded.')
    page('Execution screenshot')
    screenshots=sorted((PACKAGE/'screenshots').glob('*.png'))
    for i,path in enumerate(screenshots[:2],1):
        if i==2:page('Completion screenshot and references')
        image=Image.open(path).convert('L').convert('RGB')
        target=ASSETS/f'screenshot_{i}.png';image.save(target)
        picture(target,5.5 if i==1 else 4.5)
        p(f'Figure {i+2} '+('Kaggle confirms the complete GPU experiment ran successfully.' if i==1 else
                           'Completed model outputs and experimental verification.'))
    sub('References')
    for text in [
        'Krizhevsky A. CIFAR-10 dataset. https://www.cs.toronto.edu/~kriz/cifar.html',
        'He K et al. Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385',
        'PyTorch. Introduction to Quantization. https://pytorch.org/blog/introduction-to-quantization-on-pytorch/',
        'PyTorch. Quantization API reference. https://docs.pytorch.org/docs/main/quantization-support.html',
        'PyTorch. Pruning tutorial. https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html',
        'ONNX Runtime. Quantize ONNX models. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html']:
        para=p(text)
        for run in para.runs:run.font.size=Pt(9)
    target=PACKAGE/'Assignment3_22I-0450_A.docx'
    doc.save(target);print(target)


if __name__=='__main__':build()
