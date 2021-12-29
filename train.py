from utils import *
from data import *
import torch

data_path = 'C:\\Users\\Dutchfoundation\\Desktop\\Research\\Saudia_research\\total_data.npy'

total_data = data_pred(data_path)
data_set = DataPrep(total_data)

train_set, val_set = random_split(data_set, [800, 30])

train_loader = DataLoader(train_set, batch_size = 5, shuffle = True, num_workers = 0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size = 5, shuffle = True, num_workers = 0, pin_memory=True)


model = Network(2)
edge_model = Edge_Network(2)

model.cuda()

epochs = 100

for epoch in range(0, epochs):
  
  if epoch < 75:
    loss_function = TverskyCrossEntropyDiceWeightedLoss(2, 0.7, 0.3, 4/3, 0.8, 0.2)
    lrr = 1e-4
  if epoch >= 75:
    loss_function = TverskyCrossEntropyDiceWeightedLoss(2, 0.5, 0.5, 4/3, 0.5, 0.5)
    lrr = 1e-5

  
  optimizer = torch.optim.Adam(model.parameters(), lr=lrr, weight_decay=1e-3)
  edge_optimizer = torch.optim.Adam(edge_model.parameters(), lr=lrr, weight_decay=1e-3)

  total_train_loss = 0
  total_train_miou = 0
  train_loss = 0
  train_miou_average = 0
  train_average = 0
  train_count = 0
  total_val_miou = 0
  val_average = 0
  val_count = 0
  train_miou = 0
  val_miou = 0
  train_class_iou = 0
  total_train_class_iou = 0
  val_class_iou = 0
  total_val_class_iou = 0


  for sample in train_loader:
    model.train()
    train_x, train_y, train_edge = sample
    train_x, train_y, train_edge = train_x.cuda(), train_y.cuda(), train_edge.cuda()

    optimizer.zero_grad()
    edge_optimizer.zero_grad()

    mask, x = model(train_x)
    loss = loss_function(mask, train_y)

    x = x.detach()
    mask = mask.detach()
    edge = edge_model(x, mask)

    edge_loss = loss_function(edge, train_edge)

    train_loss += loss.item()

    mask = torch.argmax(mask.squeeze(), dim=1).detach().cpu().numpy()

    train_y = train_y.squeeze().detach().cpu()

    tmiou, ciou = mIoU(mask, train_y, 2)

    train_miou += tmiou
    train_class_iou += ciou

    loss.backward()
    optimizer.step()
    edge_loss.backward()
    edge_optimizer.step()

    total_train_loss += train_loss
    total_train_miou += train_miou

    train_loss = 0
    train_miou = 0
    train_count += 1

  train_average = total_train_loss / train_count
  total_train_class_iou = train_class_iou / train_count
  train_miou_average = total_train_miou/train_count

  
  for sample in val_loader:
    model.eval()
    val_x, val_y, val_edge= sample
    val_x, val_y = val_x.cuda(), val_y.cuda(), val_edge.cuda()

    with torch.no_grad():
      mask, x = model(val_x)
      edge = edge_model(x, mask)
      
    mask = torch.argmax(mask.squeeze(), dim=1).detach().cpu().numpy()
    val_y = val_y.squeeze().detach().cpu()

    tmiou, ciou = mIoU(mask, val_y, 2)
    val_miou += tmiou
    val_class_iou += ciou

    total_val_miou += val_miou
    val_miou = 0
    val_count += 1
  val_average = total_val_miou / val_count
  total_val_class_iou = val_class_iou / val_count
  
  print('\n',f"Epoch: {epoch+1},Training_loss: {train_average}, Train_mIoU: {train_miou_average*100},  Val_mIoU: {val_average*100}")
  print(f"Train_class_IoU: {total_train_class_iou*100},  Val_class_IoU: {total_val_class_iou*100}")


  if (epoch + 1)%1 == 0:
    torch.save(model.state_dict(),"/content/drive/MyDrive/HED_UNET{}.pth".format(9999))
    print('          MODEL SAVED          ')