import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions


# Running tests
opt = TestOptions().parse(print_options=False)
model_path = opt.model_path
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]
# try:
# dataroot = os.path.join(opt.dataroot, 'test')
# vals = os.listdir(os.path.join(opt.dataroot, 'test'))
# except:
dataroot = opt.dataroot
vals = os.listdir(opt.dataroot)
print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    # opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.classes = [''] if '0_real' in os.listdir(opt.dataroot) else os.listdir(opt.dataroot)
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, r_acc, f_acc, _, _ = validate(model, opt)
    rows.append([val, acc, ap])
    print("({}) acc: {}; ap: {}".format(val, acc, ap))

csv_name = os.path.join('./results', f'/{model_name}.csv')
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
