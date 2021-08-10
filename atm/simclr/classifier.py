import torch.nn as nn

class SimCLRClassifier(nn.Module):
   def __init__(self, n_classes, freeze_base, embeddings_model_path, hidden_size=512):
       super().__init__()
      
       base_model = ImageEmbeddingModule.load_from_checkpoint(embeddings_model_path).model
      
       self.embeddings = base_model.embedding
      
       if freeze_base:
           print("Freezing embeddings")
           for param in self.embeddings.parameters():
               param.requires_grad = False
              
       self.classifier = nn.Linear(in_features=base_model.projection[0].in_features,
                     out_features=n_classes if n_classes > 2 else 1)
  
   def forward(self, X, *args):
       emb = self.embeddings(X)
       return self.classifier(emb)

class SimCLRClassifierModule(pl.LightningModule):
   def __init__(self, hparams):
       super().__init__()
       hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
       self.hparams = hparams
       self.model = SimCLRClassifier(hparams.n_classes, hparams.freeze_base,
                                     hparams.embeddings_path,
                                     self.hparams.hidden_size)
       self.loss = nn.CrossEntropyLoss()
  
   def total_steps(self):
       return len(self.train_dataloader()) // self.hparams.epochs
  
   def preprocessing(seff):
       return transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ])
  
   def get_dataloader(self, split):
       return DataLoader(STL10(".", split=split, transform=self.preprocessing()),
                         batch_size=self.hparams.batch_size,
                         shuffle=split=="train",
                         num_workers=cpu_count(),
                        drop_last=False)
  
   def train_dataloader(self):
       return self.get_dataloader("train")
  
   def val_dataloader(self):
       return self.get_dataloader("test")
  
   def forward(self, X):
       return self.model(X)
  
   def step(self, batch, step_name = "train"):
       X, y = batch
       y_out = self.forward(X)
       loss = self.loss(y_out, y)
       loss_key = f"{step_name}_loss"
       tensorboard_logs = {loss_key: loss}

       return { ("loss" if step_name == "train" else loss_key): loss, 'log':
tensorboard_logs,
                       "progress_bar": {loss_key: loss}}
  
   def training_step(self, batch, batch_idx):
       return self.step(batch, "train")
  
   def validation_step(self, batch, batch_idx):
       return self.step(batch, "val")
  
   def test_step(self, batch, batch_idx):
       return self.step(Batch, "test")
  
   def validation_end(self, outputs):
       if len(outputs) == 0:
           return {"val_loss": torch.tensor(0)}
       else:
           loss = torch.stack([x["val_loss"] for x in outputs]).mean()
           return {"val_loss": loss, "log": {"val_loss": loss}}

   def configure_optimizers(self):
       optimizer = RMSprop(self.model.parameters(), lr=self.hparams.lr)
       schedulers = [
           CosineAnnealingLR(optimizer, self.hparams.epochs)
       ] if self.hparams.epochs > 1 else []
       return [optimizer], schedulers

hparams_cls = Namespace(
   lr=1e-3,
   epochs=5,
   batch_size=160,
   n_classes=10,
   freeze_base=True,
   embeddings_path="./efficientnet-b0-stl10-embeddings.ckpt",
   hidden_size=512
)
module = SimCLRClassifierModule(hparams_cls)
trainer = pl.Trainer(gpus=1, max_epochs=hparams_cls.epochs)
trainer.fit(module)