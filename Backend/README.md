# Здесь расположен контейнер с алгоритмом классификации

## Сборка

Так как пока в registry я его не добавил, собирать локально так

```bash

#из корня
cd Backend
docker build -t stub-service-fastapi .

```

## Запуск 

Запускаем локально контейнер так


```bash

docker run -p 5000:5000 -v /path/to/outputs:/outputs -v /path/to/inputs:/inputs --name stub-service-container stub-service-fastapi

```
Если нужно запустить в фоновом режиме, добавить флаг -d

## Запрос

тестим запросы так


```bash

curl -X POST -H "Content-Type: application/json" -d '{"input_file": "/inputs/test_input.json", "output_dir": "/outputs"}' http://0.0.0.0:5000/process

```