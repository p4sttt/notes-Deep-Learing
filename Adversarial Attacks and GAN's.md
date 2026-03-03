---
title: "Adversarial Attacks and GAN's"
slug: "adversarial-attacks-and-gan"
date: 2026-03-01
draft: false
---

# Adversarial attacks

## Идея

Рассмотрим обученную нейросеть $f(x)$ для задачи классификации.  
Пусть $x \in \mathbb{R}^d$ — вход (например, изображение), $y_{\text{true}}$ — истинная метка, $\mathcal{L}(f(x), y)$ — функция потерь.

Интуитивная цель adversarial-атаки — найти такой вход $\tilde{x}$, который:
1. Почти не отличается от исходного $x$,
2. Но приводит к неверному предсказанию модели.

То есть мы хотим «чуть-чуть» изменить вход, чтобы модель ошиблась.

## Формализация задачи

Мы ищем возмущение $\delta$, такое что:

$$
\tilde{x} = x + \delta,
$$

и выполняются условия:
1. Ограничение на размер возмущения:

$$
\|\delta\|_p \le \varepsilon,
$$

2. Ошибка классификации:

$$
f(\tilde{x}) \ne y_{\text{true}}.
$$

Обычно используют норму $\ell_\infty$:

$$
\|\delta\|_\infty = \max_i |\delta_i|.
$$

Это означает, что каждый пиксель можно изменить не более чем на $\varepsilon$.  Величину $\varepsilon$ называют **бюджетом атаки**.

## Классификация атак

### White-box атаки

Атакующий:
- знает архитектуру модели,
- знает параметры,
- имеет доступ к градиентам.

Можно использовать $\nabla_x \mathcal{L}$.

### Black-box атаки

Нет доступа к градиентам.  
Можно:
- использовать приближённые градиенты,
- обучать surrogate-модель,
- применять transferability атак.

### Untargeted

Хотим просто добиться ошибки:

$$
\max_{\|\delta\|\le \varepsilon}
\mathcal{L}(f(x+\delta), y_{\text{true}})
$$

Мы увеличиваем loss.

### Targeted

Хотим, чтобы модель выдала конкретный класс $y_{\text{target}}$:

$$
\min_{\|\delta\|\le \varepsilon}
\mathcal{L}(f(x+\delta), y_{\text{target}})
$$

Здесь мы минимизируем loss относительно целевого класса.

## FGSM (Fast Gradient Sign Method)

### Интуиция

Если loss локально почти линейный, то:

$$
\mathcal{L}(x+\delta) \approx 
\mathcal{L}(x) + \nabla_x \mathcal{L}(x)^T \delta.
$$

Чтобы максимально увеличить loss при ограничении $\|\delta\|_\infty \le \varepsilon$, нужно взять:

$$
\delta = \varepsilon \cdot \operatorname{sign}(\nabla_x \mathcal{L}(f(x), y)).
$$

### Untargeted FGSM

$$
\tilde{x} = x + \varepsilon \cdot 
\operatorname{sign}(\nabla_x \mathcal{L}(f(x), y_{\text{true}})).
$$


### Targeted FGSM

Теперь нужно уменьшать loss относительно целевого класса:

$$
\tilde{x} = x - \varepsilon \cdot 
\operatorname{sign}(\nabla_x \mathcal{L}(f(x), y_{\text{target}})).
$$

Знак меняется, потому что мы минимизируем

## Iterative FGSM (I-FGSM)

Одношаговая атака может быть недостаточно сильной.

Пусть:

$$
\tilde{x}^{(0)} = x.
$$

Итерации:

$$
\tilde{x}^{(k+1)} =
\text{Clip}_{x,\varepsilon}
\Big(
\tilde{x}^{(k)} + \alpha 
\operatorname{sign}(\nabla_x \mathcal{L}(f(\tilde{x}^{(k)}), y))
\Big).
$$

где:
- $\alpha = \varepsilon / N$,
- $N$ — число шагов,
- Clip гарантирует $\|\tilde{x} - x\|_\infty \le \varepsilon$.

Итеративные методы обычно сильнее одношаговых.

---

# Generative Adversarial Networks (GAN)

## Интуитивная идея

Хотим генерировать данные, похожие на реальные.

Автоэнкодеры тоже генерируют данные, но:
- они детерминированы,
- плохо масштабируются для сложных распределений,
- неявно моделируют распределение.

GAN предлагает другой подход: два игрока в состязательной игре

## Архитектура

1. **Генератор** $G(z)$  
   - вход: шум $z \sim p_z(z)$,
   - выход: изображение $G(z)$.
1. **Дискриминатор** $D(x)$  
   - возвращает вероятность того, что $x$ — реальное изображение.

Генератор никогда не видит реальные изображения напрямую — он получает только градиенты от дискриминатора.

## Формализация GAN

Пусть:
- $p_{\text{data}}(x)$ — распределение реальных данных,
- $p_g(x)$ — распределение, индуцированное генератором.

Целевая функция:

$$
\min_G \max_D V(D,G),
$$

где

$$
V(D,G) =
\mathbb{E}_{x \sim p_{\text{data}}}
[\log D(x)]+\mathbb{E}_{z \sim p_z}
[\log (1 - D(G(z)))].
$$

## Оптимальный дискриминатор

Зафиксируем $G$.

Тогда задача по $D$:

$$
\max_D
\int
p_{\text{data}}(x)\log D(x)+p_g(x)\log(1-D(x))
\,dx.
$$

Берём производную по $D(x)$:

$$
\frac{p_{\text{data}}(x)}{D(x)}-\frac{p_g(x)}{1-D(x)} = 0.
$$

Отсюда:

$$
D^*(x) =\frac{p_{\text{data}}(x)}
{p_{\text{data}}(x)+p_g(x)}.
$$

## Подстановка в функционал

Подставляя $D^*$ в $V(D,G)$, получаем:

$$
V(G) =- \log 4+2 \cdot\mathrm{JS}(p_{\text{data}} \| p_g).
$$

где $\mathrm{JS}$ — дивергенция Йенсена–Шеннона:

$$
\mathrm{JS}(P\|Q)=
\frac{1}{2}\mathrm{KL}(P\|M)+
\frac{1}{2}\mathrm{KL}(Q\|M),
M = \frac{1}{2}(P+Q).
$$

Следовательно, генератор минимизирует:

$$
\mathrm{JS}(p_{\text{data}} \| p_g).
$$

Таким образом, формализм полностью соответствует интуиции:  
мы подгоняем распределение генератора к распределению данных.

## Проблемы vanilla GAN

1. Нестабильное обучение.
2. Saturation градиентов.
3. Mode collapse.
4. Чувствительность к гиперпараметрам.

## Основные улучшения

### 1. Non-saturating loss

Вместо минимизации

$$
\mathbb{E}[\log(1-D(G(z)))]
$$

используют:

$$
\max_G\mathbb{E}[\log D(G(z))].
$$

### 2. WGAN

Минимизирует расстояние Вассерштейна:

$$
W(p_{\text{data}}, p_g).
$$

Требуется условие Липшица:

$$
\| \nabla_x D(x) \| \le 1.
$$

### 3. Conditional GAN (cGAN)

Добавляем условие $c$:

$$
G(z, c), \quad D(x, c).
$$

Функционал:

$$
\mathbb{E}_{x \sim p_{\text{data}}}
[\log D(x,c)]+\mathbb{E}_{z}
[\log(1-D(G(z,c),c))].
$$

Можно добавлять дополнительный классификационный loss

### 4. StyleGAN

- стиль вводится через промежуточное латентное пространство
- используется adaptive instance normalization
- достигается высокая управляемость генерации

---

# Ссылки (arXiv)

Adversarial attacks:
- [Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, 2014](https://arxiv.org/abs/1412.6572)
- [Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, 2017](https://arxiv.org/abs/1706.06083)

GAN:
- [Goodfellow et al., *Generative Adversarial Nets*, 2014](https://arxiv.org/abs/1406.2661)
- [Arjovsky et al., *Wasserstein GAN*, 2017](https://arxiv.org/abs/1701.07875)
- [Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017](https://arxiv.org/abs/1704.00028)
- [Mirza & Osindero, *Conditional GAN*, 2014](https://arxiv.org/abs/1411.1784)
- [Karras et al., *A Style-Based Generator Architecture for GANs*, 2018](https://arxiv.org/abs/1812.04948)