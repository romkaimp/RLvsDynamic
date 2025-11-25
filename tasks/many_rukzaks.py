import numpy as np
from enum import Enum
from typing import Tuple, List


class DistributionType(Enum):
    NORMAL = "normal"  # mu, sigma
    CAUCHY = "cauchy"  # x0, gamma
    UNIFORM = "uniform"  # a, b

    def get_parameters(self) -> Tuple[str, ...]:
        """Возвращает названия параметров для распределения"""
        return {
            DistributionType.NORMAL: ("mu", "sigma"),
            DistributionType.CAUCHY: ("x0", "gamma"),
            DistributionType.UNIFORM: ("a", "b")
        }[self]

    def get_default_params(self) -> Tuple[float, ...]:
        """Возвращает параметры по умолчанию"""
        return {
            DistributionType.NORMAL: (0.0, 1.0),  # mu=0, sigma=1
            DistributionType.CAUCHY: (0.0, 1.0),  # x0=0, gamma=1
            DistributionType.UNIFORM: (0.0, 1.0)  # a=0, b=1
        }[self]

    def generate_samples(self, size: int, **params) -> np.ndarray:
        """Генерирует samples из распределения с заданными параметрами"""
        default_params = dict(zip(self.get_parameters(), self.get_default_params()))
        actual_params = {**default_params, **params}

        if self == DistributionType.NORMAL:
            mu = actual_params["mu"]
            sigma = actual_params["sigma"]
            return np.random.normal(mu, sigma, size)
        elif self == DistributionType.CAUCHY:
            x0 = actual_params["x0"]
            gamma = actual_params["gamma"]
            return np.random.standard_cauchy(size) * gamma + x0
        elif self == DistributionType.UNIFORM:
            a = actual_params["a"]
            b = actual_params["b"]
            return np.random.uniform(a, b, size)


class Task:
    def __init__(self, n: int, k: int, distrs: Tuple[DistributionType, DistributionType, DistributionType],
                 volume_params: dict = None, cost_params: dict = None, item_volume_params: dict = None):
        """
        Генерирует задачу о нескольких рюкзаках

        Args:
            n: количество рюкзаков
            k: количество предметов
            distrs: кортеж из трех DistributionType для:
                   [0] - распределение вместимости рюкзаков
                   [1] - распределение стоимости предметов
                   [2] - распределение размеров предметов
            volume_params: параметры для распределения вместимости рюкзаков
            cost_params: параметры для распределения стоимости предметов
            item_volume_params: параметры для распределения размеров предметов
        """
        self.n = n
        self.k = k
        self.distributions = distrs

        # Параметры по умолчанию
        default_volume_params = {'mu': 50, 'sigma': 10} if distrs[0] == DistributionType.NORMAL else \
            {'a': 30, 'b': 70} if distrs[0] == DistributionType.UNIFORM else \
                {'x0': 50, 'gamma': 5}

        default_cost_params = {'mu': 20, 'sigma': 5} if distrs[1] == DistributionType.NORMAL else \
            {'a': 10, 'b': 30} if distrs[1] == DistributionType.UNIFORM else \
                {'x0': 20, 'gamma': 3}

        default_item_volume_params = {'mu': 10, 'sigma': 3} if distrs[2] == DistributionType.NORMAL else \
            {'a': 5, 'b': 15} if distrs[2] == DistributionType.UNIFORM else \
                {'x0': 10, 'gamma': 2}

        # Объединяем с пользовательскими параметрами
        volume_params = volume_params or default_volume_params
        cost_params = cost_params or default_cost_params
        item_volume_params = item_volume_params or default_item_volume_params

        # Генерация данных
        self.rukzaks_volume = self._generate_positive_values(distrs[0], n, **volume_params)
        self.items_cost = self._generate_positive_values(distrs[1], k, **cost_params)
        self.items_volume = self._generate_positive_values(distrs[2], k, **item_volume_params)

        # Проверка корректности данных
        self._validate_data()
        self.total_sum = 0

    def _generate_positive_values(self, dist: DistributionType, size: int, **params) -> np.ndarray:
        """Генерирует положительные значения из распределения"""
        samples = dist.generate_samples(size, **params)

        # Обеспечиваем положительные значения (берем модуль и добавляем маленькое значение)
        positive_samples = np.abs(samples)

        # Если все значения близки к нулю, масштабируем
        if np.max(positive_samples) < 1e-6:
            positive_samples = positive_samples * 100 + 1

        return positive_samples

    def _validate_data(self):
        """Проверяет корректность сгенерированных данных"""
        assert len(self.rukzaks_volume) == self.n, "Неверное количество рюкзаков"
        assert len(self.items_cost) == self.k, "Неверное количество стоимостей предметов"
        assert len(self.items_volume) == self.k, "Неверное количество размеров предметов"

        assert np.all(self.rukzaks_volume > 0), "Вместимости рюкзаков должны быть положительными"
        assert np.all(self.items_cost > 0), "Стоимости предметов должны быть положительными"
        assert np.all(self.items_volume > 0), "Размеры предметов должны быть положительными"

    def __str__(self):
        return (f"Task(n={self.n}, k={self.k})\n"
                f"Рюкзаки: {self.rukzaks_volume}\n"
                f"Стоимости: {self.items_cost}\n"
                f"Размеры: {self.items_volume}\n"
                f"Распределения: {[dist.name for dist in self.distributions]}")

    def get_state(self):
        return [self.rukzaks_volume, self.items_volume, self.items_cost]

    def take_action(self, action_1, action_2):
        cont = True
        if self.rukzaks_volume[action_1] < self.items_volume[action_2]:
            cont = False
            return cont, -5
        self.rukzaks_volume[action_1] -= self.items_volume[action_2]
        self.total_sum += self.items_cost[action_2]
        rew = self.items_cost[action_2]
        self.items_cost = np.delete(self.items_cost, action_2)
        self.items_volume = np.delete(self.items_volume, action_2)

        return cont, rew


    def get_total_capacity(self) -> float:
        """Общая вместимость всех рюкзаков"""
        return np.sum(self.rukzaks_volume)

    def get_total_item_volume(self) -> float:
        """Общий объем всех предметов"""
        return np.sum(self.items_volume)

    def is_feasible(self) -> bool:
        """Проверяет, помещаются ли все предметы в рюкзаки"""
        return self.get_total_item_volume() <= self.get_total_capacity()

    def not_end(self):
        min_volume = min(self.items_volume)
        max_rukzak = max(self.rukzaks_volume)
        return min_volume <= max_rukzak

    def solve_dynamically(self) -> bool:
        total_sum = 0
        self.proportion = self.items_cost / self.items_volume

        # Получаем индексы для сортировки по убыванию пропорций
        sorted_indices = np.argsort(self.proportion)[::-1]  # [::-1] для убывания

        # Сортируем все массивы по этим индексам
        # proportion = self.proportion[sorted_indices]
        items_cost = self.items_cost[sorted_indices]
        items_volume = self.items_volume[sorted_indices]

        counter = 0
        for volume in self.rukzaks_volume:
            total_volume = 0
            while counter < self.k and total_volume + items_volume[counter] <= volume:
                total_sum += items_cost[counter]
                total_volume += items_volume[counter]
                counter += 1
        return total_sum

    # def solve_dynamically(self):
    #     pass


# Примеры использования
if __name__ == "__main__":
    # Пример 1: Все равномерные распределения
    task1 = Task(
        n=3,
        k=100,
        distrs=(DistributionType.UNIFORM, DistributionType.UNIFORM, DistributionType.UNIFORM),
        volume_params={'a': 40, 'b': 60},
        cost_params={'a': 15, 'b': 25},
        item_volume_params={'a': 8, 'b': 12}
    )
    print("Пример 1:")
    print(task1)
    print(f"Общая вместимость: {task1.get_total_capacity():.2f}")
    print(f"Общий объем предметов: {task1.get_total_item_volume():.2f}")
    print(f"Задача выполнима: {task1.is_feasible()}")
    print(task1.solve_dynamically())

    # Пример 2: Нормальное распределение для всего
    task2 = Task(
        n=4,
        k=8,
        distrs=(DistributionType.NORMAL, DistributionType.NORMAL, DistributionType.NORMAL),
        volume_params={'mu': 50, 'sigma': 8},
        cost_params={'mu': 20, 'sigma': 4},
        item_volume_params={'mu': 12, 'sigma': 2}
    )
    print("Пример 2:")
    print(task2)
    print()

    # Пример 3: Смешанные распределения
    task3 = Task(
        n=2,
        k=5,
        distrs=(DistributionType.UNIFORM, DistributionType.NORMAL, DistributionType.CAUCHY),
        volume_params={'a': 30, 'b': 50},
        cost_params={'mu': 25, 'sigma': 5},
        item_volume_params={'x0': 10, 'gamma': 3}
    )
    print("Пример 3:")
    print(task3)