## Запуск ML-моделей на Raspberry Pi CM4 за допомогою Coral USB Accelerator

#### Короткий список інструкцій
 - Даунгрейд python до версії 3.9 (остання підтримувана версія для бібліотеки PyCoral)
 - Встановлення залежностей (wheels для tflite, pycoral)
 - Підготовка енву (Edge TPU runtime) і підключення Coral USB Accelerator до малинки
 - Встановлення залежностей для ML-коду (залежності pip для pillow, і т.д.)
 - Запуск програми

Для підготовки енву треба запустити:
```. coral-setup.sh ```

Для запуску моделі з захардкодженим списком зображень:
```python run.py```

Для запуску моделі з вашим URL - додайте URL першим параметром скрипта:
```python run.py <URL_HERE>```

#### Ці команди є в скрипті, але бажано їх додати в .bashrc для зручності дебагу/тестування:
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```