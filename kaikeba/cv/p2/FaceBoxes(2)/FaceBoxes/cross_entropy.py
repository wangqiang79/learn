import torch

format_str = '%%-%ds' % 15

def neg_log_softmax_v1(x):

    print('\nv1')
    exp = torch.exp(x)
    print(format_str % 'exp', exp)
    sum_exp = torch.sum(exp)
    print(format_str % 'sum_exp', sum_exp)
    prob = exp / sum_exp
    print(format_str % 'prob', prob)
    neg_log_prob = -torch.log(prob)
    print(format_str % 'neg_log_prob', neg_log_prob)


def neg_log_softmax_v2(x):

    print('\nv2')
    exp = torch.exp(x)
    print(format_str % 'exp', exp)
    sum_exp = torch.sum(exp)
    print(format_str % 'sum_exp', sum_exp)
    log_sum_exp = torch.log(sum_exp)
    print(format_str % 'log_sum_exp', log_sum_exp)
    neg_log_prob = log_sum_exp - x
    print(format_str % 'neg_log_prob', neg_log_prob)


def neg_log_softmax_v3(x):

    print('\nv3')
    x_sub = x - x.max()
    print(format_str % 'x_sub', x_sub)
    exp_sub = torch.exp(x_sub)
    print(format_str % 'exp_sub', exp_sub)
    sum_exp_sub = torch.sum(exp_sub)
    print(format_str % 'sum_exp_sub', sum_exp_sub)
    log_sum_exp_sub = torch.log(sum_exp_sub)
    print(format_str % 'log_sum_exp_sub', log_sum_exp_sub)
    log_sum_exp = log_sum_exp_sub + x.max()
    print(format_str % 'log_sum_exp', log_sum_exp)
    neg_log_prob = log_sum_exp - x
    print(format_str % 'neg_log_prob', neg_log_prob)


if __name__ == '__main__':

    # x = torch.tensor([-10, 2.3, 1.2, 80, 0.5])       # v1/v2/v3 can work well
    # x = torch.tensor([-100, 2.3, 1.2, 80, 0.5])      # v2/v3 can work well
    # x = torch.tensor([-100, 2.3, 1.2, 90, 0.5])      # v3 can work well
    x = torch.tensor([-1e6, -1e6, -1e6, -1e6, -1e6]) # v3 can work well
    print(format_str % 'x', x)

    neg_log_softmax_v1(x)
    neg_log_softmax_v2(x)
    neg_log_softmax_v3(x)
