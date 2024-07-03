import torch

def get_optim(model, policy_net, args):
    criterion = torch.nn.CrossEntropyLoss()
    
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(policy_net.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(policy_net.parameters()), lr=args.lr)
        
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    return criterion, optimizer, main_lr_scheduler
    