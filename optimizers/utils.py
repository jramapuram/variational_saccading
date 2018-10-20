def decay_lr_every(optimizer, lr, epoch, decay_every=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# TODO: noam scheduler
# "Schedule": {
#     "name": "noam_learning_rate_decay",
#     "args": {
#         "warmup_steps": 4000,
#         "minimum": 1e-4
#     }
# }
