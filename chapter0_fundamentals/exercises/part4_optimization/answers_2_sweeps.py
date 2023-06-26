from answers_2 import *

MAIN = __name__ == "__main__"

if MAIN:
    cifar_trainset, cifar_testset = get_cifar(subset=1)
    cifar_trainset_small, cifar_testset_small = get_cifar(subset=10)

#%% 
if MAIN:
	sweep_config = dict(
		method = 'random',
		metric = dict(name = 'accuracy', goal = 'maximize'),
		parameters = dict(
			batch_size = dict(values = [32, 64, 128, 256]),
			max_epochs = dict(min = 1, max = 4),
			learning_rate = dict(max = 0.1, min = 0.0001, distribution = 'log_uniform_values'),
		)
	)
	tests.test_sweep_config(sweep_config)

def train():
    args = ResNetFinetuningArgsWandb(trainset=cifar_trainset_small, testset=cifar_testset_small)
    args.batch_size=wandb.config["batch_size"]
    args.max_epochs=wandb.config["max_epochs"]
    args.learning_rate=wandb.config["learning_rate"]

    model = LitResNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=args.logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.testloader)
    wandb.finish()

# %%
if MAIN:
    sweep_id = wandb.sweep(sweep=sweep_config, project='day4-resnet-sweep')
    wandb.agent(sweep_id=sweep_id, function=train, count=5)
    '''
    Best:
        Key
        Value
        batch_size
        256
        learning_rate
        0.0024223055861088525
        max_epochs
        2
    '''