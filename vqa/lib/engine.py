import time
import torch
from torch.autograd import Variable
import vqa.lib.utils as utils
from torchsummary import summary
import torchvision
from torch import nn

def train(loader, model, criterion, optimizer, logger, epoch, print_freq=10):
    # switch to train mode
    model.train()
    meters = logger.reset_meters('train')

    resnet = torchvision.models.resnet152(pretrained=True)
    base_feat = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,resnet.maxpool,resnet.layer1,resnet.layer2,
    resnet.layer3,resnet.layer4).cuda()
    base_feat = nn.DataParallel(base_feat)

    end = time.time()
    for i, sample in enumerate(loader):
        # batch_size = sample['visual'].size(0)
        batch_size = sample[0].size(0)
        
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input_rgb = Variable(sample[0]).cuda()
        input_thermal = Variable(sample[1]).cuda()

        rgb_feat = base_feat(input_rgb)
        thermal_feat = base_feat(input_thermal)

        # input_visual   = Variable(sample['visual'])
        # input_question = Variable(sample['question'])
        
        # target_answer  = Variable(sample['answer'].cuda(async=True))
        # sample_question_float = sample["question"].float()
        
        # summary(model,[sample['visual'],sample_question_float])
        
        # compute output
        # output = model(input_visual, input_question)

        output = model(rgb_feat, thermal_feat)
        

        torch.cuda.synchronize()
        loss = criterion(output, target_answer)
        meters['loss'].update(loss.data[0], n=batch_size)

        # measure accuracy 
        acc1, acc5 = utils.accuracy(output.data, target_answer.data, topk=(1, 5))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})'.format(
                   epoch, i, len(loader),
                   batch_time=meters['batch_time'], data_time=meters['data_time'],
                   loss=meters['loss'], acc1=meters['acc1'], acc5=meters['acc5']))

    logger.log_meters('train', n=epoch)

# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def validate(loader, model, criterion, logger, epoch=0, print_freq=10):
    results = []

    # switch to evaluate mode
    model.eval()
    meters = logger.reset_meters('val')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        input_visual   = Variable(sample['visual'].cuda(async=True), volatile=True)
        input_question = Variable(sample['question'].cuda(async=True), volatile=True)
        target_answer  = Variable(sample['answer'].cuda(async=True), volatile=True)

        # compute output
        output = model(input_visual, input_question)
        loss = criterion(output, target_answer)
        meters['loss'].update(loss.data[0], n=batch_size)

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output.data, target_answer.data, topk=(1, 5))
        meters['acc1'].update(acc1[0], n=batch_size)
        meters['acc5'].update(acc5[0], n=batch_size)

        # compute predictions for OpenEnded accuracy
        _, pred = output.data.cpu().max(1)
        pred.squeeze_()
        for j in range(batch_size):
            results.append({'question_id': sample['question_id'][j],
                            'answer': loader.dataset.aid_to_ans[pred[j]]})

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Acc@5 {acc5.val:.3f} ({acc5.avg:.3f})'.format(
                   i, len(loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], loss=meters['loss'],
                   acc1=meters['acc1'], acc5=meters['acc5']))

    print(' * Acc@1 {acc1.avg:.3f} Acc@5 {acc5.avg:.3f}'
          .format(acc1=meters['acc1'], acc5=meters['acc1']))

    logger.log_meters('val', n=epoch)
    return meters['acc1'].avg, results


def test(loader, model, logger, epoch=0, print_freq=10):
    results = []
    testdev_results = []

    model.eval()
    meters = logger.reset_meters('test')

    end = time.time()
    for i, sample in enumerate(loader):
        batch_size = sample['visual'].size(0)
        input_visual   = Variable(sample['visual'].cuda(async=True), volatile=True)
        input_question = Variable(sample['question'].cuda(async=True), volatile=True)

        # compute output
        output = model(input_visual, input_question)

        # compute predictions for OpenEnded accuracy
        _, pred = output.data.cpu().max(1)
        pred.squeeze_()
        for j in range(batch_size):
            item = {'question_id': sample['question_id'][j],
                    'answer': loader.dataset.aid_to_ans[pred[j]]}
            results.append(item)
            if sample['is_testdev'][j]:
                testdev_results.append(item)

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                   i, len(loader), batch_time=meters['batch_time']))

    logger.log_meters('test', n=epoch)
    return results, testdev_results
