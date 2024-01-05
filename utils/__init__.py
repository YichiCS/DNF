
def result(map_result, acc_result, label_acc_result, writer=None, epoch=-1):
    r_acc = format(label_acc_result[0].item(), ".4f")
    f_acc = format(label_acc_result[1].item(), ".4f")
    acc = format(acc_result.item(), ".4f")
    map = format(map_result.item(), ".4f")

    print(f'Epoch {epoch}: Acc = {acc} | R_Acc = {r_acc} | F_Acc = {f_acc} | mAP: {map}')

    if writer is not None:
        writer.writerow([epoch, acc, r_acc, f_acc, map])   