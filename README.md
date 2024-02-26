# HalfCheetah

Имплементация алгоритмов DDPG и PPO. 

Обучение и тестирование в среде [Half-Cheetah](https://robotics.farama.org/envs/MaMuJoCo/ma_half_cheetah/) из MuJoCo.

Статья про PPO, по которой писался алгоритм: https://arxiv.org/pdf/1707.06347.pdf

# Запуск
Для запуска обучения алгоритма DDPG:
```python
python3 train_ddpg.py
```

Для запуска обучения алгоритма PPO:
```python
python3 train_ppo.py
```

# Результат
На данный момент удалось достичь такого результата с помощью алгоритма PPO. Требуется доработка алгоритма и более тщательный подбор гиперпараметров.

https://github.com/sadevans/HalfCheetah/assets/82286355/d97b2ffa-85f7-495f-af06-2864927792f4




# Дальнейшее развитие
- TRPO
- SAC
- A3C
