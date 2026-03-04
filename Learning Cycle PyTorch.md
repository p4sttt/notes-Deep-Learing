---
title: "Learning cycle in PyTorch"
slug: "learning-cycle-pytorch"
date: 2026-03-04
draft: false
---

# Введение

Идейно цикл обучения нейросети выглядит следующим образом:
1. Получение батча данных из датасета  
2. Вычисление нейронной сети на батче данных  
3. Вычисление функции ошибки  
4. Расчет градиентов с помощью обратного распространения ошибки  
5. Шаг оптимизации  
6. Переход на пункт 1  

# Работы с данными

Существует классический датасет ImageNet для обучения классификации изображений. Проблема в том, что входные картинки могут быть разного размера. Как быть?

Обычно изображения приводят к одному размеру с помощью трансформаций. Это можно сделать несколькими способами:

- `Resize` — масштабировать изображение до фиксированного размера  
- `CenterCrop` или `RandomCrop` — вырезать фрагмент нужного размера  
- `Pad` — дополнить изображение до нужного размера  

Чаще всего используется комбинация `Resize + Crop`. Это позволяет привести все входные изображения к одинаковой размерности, необходимой для батчевой обработки.

Для работы с датасетом нам следует обернуть данные в `torch.utils.data.Dataset`, так как это удобная обертка для данных, которая унифицирует доступ к ним. Использование `torch.utils.data.Dataset` позволяет удобно и эффективно итерироваться и получать доступ к данным. Класс-наследник должен переопределять магические методы `__len__` и `__getitem__`.

Простой пример реализации:

```python
import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, paths, targets, transform=None):
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx]
```

Advanced-фичи, которые часто используют в проде:

- **кеширование** изображений в памяти  
- **предварительная индексация** путей и метаданных  
- **memory-mapping** больших файлов  
- **lazy loading** (загрузка только в момент обращения)  
- **обработка ошибок чтения** (битые файлы)  
- **балансировка классов** внутри `Dataset`

Кроме того, одной из серьезных проблем в машинном обучении является переобучение. Для борьбы с ним обычно используют аугментацию и трансформацию данных. Обычно все аугментации и трансформации применяются к изображению при обращении к элементу датасета. Такой подход позволяет не хранить аугментированную выборку, что очень важно в случае непрерывных изменений (повороты, изменение контраста, яркости и тому подобное). Нейронная сеть, оптимизация и другие абстракции внутри цикла обучения (кроме датасета) не должны зависеть от аугментации данных.

Дополнительный пример использования трансформаций:

```python
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
```

```Python
form torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
	def __init__(self, data_df, is_train, transform):
		super().__init__();
		mask = data_df['is_valid'] != is_train
		
		self.img_paths = data_df[mask].path.values
		self.targets = data_df[mask].target.values
		
		self.transform = transform
 		
	def __getitem__(self, idx):
		path = self.img_paths[idx]
		img_path = os.path.join(DATA_DIR, path)
		
		img = Image.open(img_path).convert('RGB')
		img = self.transform(img)
		target = self.targets[idx]
		
		return img, target
		
	def __len__(self):
		return len(self.targets)
```

Дорогая ли операция `__getitem__`? Для получения картинки мы обращаемся к диску, и это действительно медленно. Улучшить ситуацию можно несколькими способами:

- использовать **несколько процессов загрузки** (`num_workers` в `DataLoader`)  
- хранить данные на **быстром SSD**  
- применять **кеширование**  
- хранить данные в **LMDB / WebDataset / TFRecord**  
- предварительно **декодировать изображения**

Для получения батча мы воспользуемся `torch.utils.data.DataLoader`, который позволяет быстро получать батчи данных. Реализация `DataLoader` — достаточно нетривиальный функционал. Подробно все параметры можно прочитать в документации, рассмотрим базовые.

Основные аргументы конструктора:

- `dataset` — датасет  
- `batch_size` — размер батча  
- `shuffle` — перемешивание данных  
- `sampler` — пользовательский способ выборки  
- `batch_sampler` — выборка батчей  
- `num_workers` — число процессов загрузки данных  
- `collate_fn` — функция сборки батча  
- `pin_memory` — ускорение передачи на GPU  
- `drop_last` — отбросить последний неполный батч  
- `timeout` — таймаут загрузки  
- `worker_init_fn` — функция инициализации воркеров  
- `persistent_workers` — не перезапускать воркеры между эпохами

Наиболее важные на практике:

- `batch_size`
- `shuffle`
- `num_workers`
- `pin_memory`
- `drop_last`

```Python
DataLoader(
	dataset,
	batch_size=1,
	shuffle=False,
	sampler=None,
	batch_sampler=None,
	num_workers=0,
	collate_fn=None,
	pin_memory=False,
	drop_last=False,
	timeout=0,
	worker_init_fn=None,
	multiprocessing_context=None,
	generator=None,
	prefetch_factor=2,
	persistent_workers=False,
	pin_memory_device=''
)
```

Пример работы с `DataLoader` с батчом 9. В качестве датасета выбран тензор размера 10. После каждой эпохи мы будем перемешивать данные и не учитывать последний батч, если он меньше остальных.

```Python
D = DataLoader(
	torch.arange(10),
	batch_size=9,
	shuffle=True,
	drop_last=True,
)

for epoch in range(5):
	for batch in D:
		print(epoch, batch)
```

В этом примере:

- `torch.arange(10)` создаёт датасет `[0,1,2,...,9]`
- `batch_size=9` означает, что батч содержит 9 элементов
- `drop_last=True` удаляет последний неполный батч (из одного элемента)
- `shuffle=True` перемешивает данные перед каждой эпохой

Поэтому на каждой эпохе мы будем получать один батч из 9 случайных элементов.

```Python
# среднее значение пикселей для i-го канала
IMG_MEAN = np.array([0.485, 0.456, 0.406])
# среднее отклонение пикселей для i-го канала
IMG_STD = np.array([0.229, 0.224, 0.225])

transform_train = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
		gransforms.RandomCrop((160, 160)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomGrayscale(p=0.1)
	]
)

transform_val = transforms.Compose(
	[
		transofrms.ToTensor(),
		transoforms.Normlize(mean=IMG_MEAN, std=IMG_STD),
		transofrms.CenterCrop((160, 160))
	]
)
```

Откуда берутся `IMG_MEAN` и `IMG_STD`? Нужно как-то нормализовать изображение. Дело в том, что математическое ожидание по каждому из каналов в реальности отличается от наших теоретических идеализированных представлений. Поэтому принято считать среднее и стандартное отклонение по обучающему датасету и нормализовать данные.

Часто используют статистики ImageNet, так как на нем предобучено большое количество моделей.

```Python
train_ds = ImageDataset(
	data_df, 
	is_train=True, 
	transform=transofrm_train
)  
valid_ds = ImageDataset(
	data_df, 
	is_train=False, 
	transform=transform_val
)
```

# Цикл обучения

С данными мы разобрались, а именно описали всё необходимое для правильной и аккуратной работы с данными: создали `Dataset` с изображениями и аугментациями, который используется для `DataLoader`. Благодаря этому мы можем не думать о работе с данными, а эффективно итерироваться по батчам.

Нам осталось определить:

- нейронную сеть — то, что будем обучать  
- оптимизатор — то, чем будем обучать  
- функцию потерь — то, чему будем учить  

После этого можно будет реализовать полноценный цикл обучения, в котором соединим всё вместе.

Воспользуемся моделью `ResNet`.

```Python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Создадим класс нейронной сети ResNet, который является наследником `torch.nn.Module`. Важно не забыть про `super().__init__()`.

```Python
class ResNet(torch.nn.Module):
	def __init__(self, n_classse):
		super().__init__()
		
		self.features = torch.nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			
			nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
			
			nn.AdaptiveAvgPool2d((1, 1))
		)
		
		self.dense = torch.nn.Sequential(
			nn.Flatten(),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(256, n_classse)
		)

		self._init_weights()
	
	def forward(self, x):
		x = self.features(x)
		x = self.dense(x)
		
		return x

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def predict(self, x):
		self.eval()
		with torch.no_grad():
			out = self.forward(x)
			return torch.argmax(out, dim=1)
```

Запустим прямой проход для случайного батча (бывает полезно убедиться, что мы правильно обработали размерности).

```Python
net = ResNet(n_classes=10)

batch = next(iter(vis_dataloader))
out = net(batch[0])

print(out.shape)
```

Посмотрим на число параметров, чтобы оценить сложность модели.

```Python
def print_params_count(model):
	total_params = sum(p.numel() for p in model.prarmeters())
	total_params_grad = sum(p.numel() for p in model.prarmeters() if p.requires_grad)
	
	mode_name = model.__class__.__name__
	print(f"{mode_name}: total params = {total_params:,}, trainable params = {total_params_grad:,}")
	
	
print_params_count(net)
```

В качестве оптимизатора возьмём `Adam`, который является стандартом в области.

Почему именно он:

- использует **адаптивные learning rates** для каждого параметра  
- сочетает идеи **Momentum** и **RMSProp**  
- хорошо работает «из коробки»  
- требует меньше настройки гиперпараметров  

```Python
LEARNING_RATE = 1e-4

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
```

Так как мы решаем задачу классификации, будем использовать метод перекрёстной энтропии `torch.nn.CrossEntropyLoss()`.

Если бы решали другие задачи:

**Регрессия**

- `MSELoss`
- `L1Loss`
- `SmoothL1Loss`

**Бинарная классификация**

- `BCELoss`
- `BCEWithLogitsLoss`

**Сегментация**

- `DiceLoss`
- `FocalLoss`

```Python
loss_fn = torch.nn.CrossEntropyLoss()
```

```Python
train_dataloader = DataLoader(
	train_ds,
	batch_size=32,
	shuffle=True,
	drop_last=True,
	num_workers=4
)

valid_dataloader = DataLoader(
	valid_ds,
	batch_size=32,
	shuffle=False,
	drop_last=False,
	num_workers=4
)
```

Наконец-то можем приступить к написанию цикла обучения (`train loop`).

```Python
def train_base(epoch_num, model, optimizer, loss_fn, train_dataloader, device):
	global_step = 0
	model.to(device)
	model.train()
	
	for _ in tqdm(range(epoch_num)):
	
		for X_batch, y_true in train_dataloader:
			optimizer.zero_grad()
			X_batch, y_true = X_batch.to(device), y_true.to(device)
			
			out = model(X_batch)
			loss = loss_fn(out, y_true)
			loss.backward()
			optimizer.step()
			
			y_pred = torch.argmax(out, 1)
			accuracy = torch.sum(y_pred == y_true) / y_pred.shape[0]
			print(f"{global_step} | loss={loss.item():0.3f} | acc={accuracy.item():0.3f}")
			
			global_step += 1
```

Хочется ещё построить график. А как это можно сделать? Если сохранять куда-то данные, то тратятся ресурсы, если перерисовывать на ходу, то для больших проектов это слишком затратно. Возникает много нюансов, которые сложно обработать.

```Python
epoch_num = 1
train_base(
	epoch_num,
	net,
	optimizer,
	loss_fn,
	train_dataloader,
	device
)
```

Для этого существует логирование. В проде используются `tensorboard`, `wandb`, `comet_ml`, `trackio`, `mlflow`.

# Логирование

```Python
import wandb
import comet_ml
```

При первом запуске потребуется указать API-ключ, его можно получить, если зарегистрироваться на официальном сайте. Для логирования кода в `wandb` достаточно указать `save_code=True`.

Пример базового логирования:

```python
import wandb

wandb.init(
    project="pytorch-training",
    name="resnet-run",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "optimizer": "Adam"
    },
    save_code=True
)

wandb.log({
    "loss": loss.item(),
    "accuracy": accuracy.item(),
    "step": global_step
})
```

Аналог для TensorBoard:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

writer.add_scalar("loss", loss.item(), global_step)
writer.add_scalar("accuracy", accuracy.item(), global_step)
```

Логирование картинок:

```python
wandb.log({
    "images": [wandb.Image(img, caption=str(label))]
})
```

Логирование гистограмм:

```python
for name, param in model.named_parameters():
    wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())})
```

# Валидация

Без валидации мы ничего не знаем о качестве модели. Важно, что на этапе валидации нам не нужно считать градиенты, а модель нужно перевести в `eval` режим.

```Python
@torch.no_grad()
def evaluate(model, valid_dataloader, loss_fn, device):
	model.to(device)
	model.eval()
	
	loss, acc = 0, 0
	count = 0
	
	for X_batch, y_true in valid_dataloader:
		X_batch, y_true = X_batch.to(device), y_true.to(device)
		
		out = model(X_batch)
		y_pred = torch.argmax(out, 1)
		
		batch_sz = out.shape[0]
		loss += loss_fn(out, y_true).item() * batch_sz
		acc += torch.sum(y_pred == y_true).item()
		count += batch_sz
		
	return loss / count, acc / count
```

```Python
def train_loger_eval(epoch_num, model, optimizer, loss_fn, train_dataloader, valid_dataloader, device):

	experiment = comet_ml.start(
		api_key=userdata.get('COMET_API_KEY'),
		project_name="pytorch-training",
		log_code=True
	)

	model.to(device)

	global_step = 0

	for epoch in range(epoch_num):

		model.train()
		train_loss = 0
		train_acc = 0
		train_count = 0

		for X_batch, y_true in train_dataloader:

			X_batch, y_true = X_batch.to(device), y_true.to(device)

			optimizer.zero_grad()

			out = model(X_batch)
			loss = loss_fn(out, y_true)

			loss.backward()
			optimizer.step()

			y_pred = torch.argmax(out, 1)

			batch_sz = out.shape[0]

			train_loss += loss.item() * batch_sz
			train_acc += torch.sum(y_pred == y_true).item()
			train_count += batch_sz

			experiment.log_metric("train_batch_loss", loss.item(), step=global_step)

			global_step += 1

		train_loss /= train_count
		train_acc /= train_count

		val_loss, val_acc = evaluate(model, valid_dataloader, loss_fn, device)

		experiment.log_metric("train_loss", train_loss, epoch=epoch)
		experiment.log_metric("train_acc", train_acc, epoch=epoch)

		experiment.log_metric("val_loss", val_loss, epoch=epoch)
		experiment.log_metric("val_acc", val_acc, epoch=epoch)

		print(
			f"epoch={epoch} | "
			f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
			f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
		)
```

Логика обучения с валидацией обычно выглядит так:

```python
for epoch in range(epoch_num):

    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)

    experiment.log_metric("train_loss", train_loss, epoch=epoch)
    experiment.log_metric("train_acc", train_acc, epoch=epoch)

    experiment.log_metric("val_loss", val_loss, epoch=epoch)
    experiment.log_metric("val_acc", val_acc, epoch=epoch)
```

# Кризис воспроизводимости

Выше мы могли заметить, что обучение сильно меняется от запуска к запуску, что создаёт следующие проблемы:

- возможно, кто-то не сможет воспроизвести ваши результаты  
- возможно, вы никогда не сможете воспроизвести свои же результаты  

Получить крутую модель, но без возможности воспроизведения — крайне плохо для индустрии. Для воспроизведения результатов достаточно просто зафиксировать `seed`. Например, ниже продемонстрирована зависимость генератора случайных чисел от устройства.

```Python
torch.manual_seed(42)
randn1 = troch.randn(10)

torch.manual_seed(42)
randn2 = troch.randn(10)

print(randn1, randn2)
```

Для начала постараемся сделать обучение независимым от очередности запуска, воспользуемся советами из официальной статьи *Reproducibility PyTorch*. Нам необходимо:

- зафиксировать seed CPU  
- зафиксировать seed GPU  
- зафиксировать seed `numpy`  
- зафиксировать seed стандартного `random`  

В целом, этого уже будет достаточно. Кроме того, существуют рандомизированные алгоритмы и алгоритмы, зависимые от модели технического оборудования. Эту случайность намного сложнее контролировать, но можно с помощью `torch.use_deterministic_algorithm(True)` — так гарантируется точная воспроизводимость, но код скорее всего замедлится.

```Python
def set_blobal_sed(seed: int) -> None:
	import random
	import numpy as np
	import torch
	
	random.seed(seed)
	np.random.seed(seed)
	
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	torch.use_deterministic_algorithms(True)
```
Пример реализации:

```python
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

# Чекпоинты модели

Во время обучения полезно сохранять промежуточные состояния модели (checkpoints). Это позволяет:

- продолжить обучение после остановки  
- выбрать лучшую модель по валидации  
- проводить эксперименты  

Пример сохранения:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch
}, "checkpoint.pt")
```

Загрузка:

```python
checkpoint = torch.load("checkpoint.pt")

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

# Ссылки

- [Репозиторий ММП ВМК МГУ](https://github.com/mmp-mmro-team)
- [Еще ссылка на репозиторий](https://github.com/mmp-practicum-team/mmp_dl_spring)