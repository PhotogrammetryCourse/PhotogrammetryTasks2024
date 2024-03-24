В этом репозитории предложены задания курса по Фотограмметрии для студентов МКН/ИТМО/ВШЭ.

[Остальные задания](https://github.com/PhotogrammetryCourse/PhotogrammetryTasks2024/).

# Задание 4. SFM. Ceres Solver и Bundle Adjustment.

[![Build Status](https://github.com/PhotogrammetryCourse/PhotogrammetryTasks2024/actions/workflows/cmake.yml/badge.svg?branch=task04&event=push)](https://github.com/PhotogrammetryCourse/PhotogrammetryTasks2024/actions/workflows/cmake.yml)

0. Установить Eigen (если не установили в прошлом задании) и  Ceres Solver - см. инструкции в CMakeLists.txt
1. Выполнить задания ниже
2. Отправить **Pull-request** с названием```Task04 <Имя> <Фамилия> <Аффиляция>```:

 - Скопируйте в описание [шаблон](https://raw.githubusercontent.com/PhotogrammetryCourse/PhotogrammetryTasks2024/task04/.github/pull_request_template.md)
 - Обязательно отправляйте PR из вашей ветки **task04** (вашего форка) в ветку **task04** (основного репозитория)
 - Перечислите свои мысли по вопросам поднятым в коде и просто появившиеся в процессе выполнения задания
 - Создайте PR
 - Затем дождавшись отработку Travis CI (около 15 минут) - скопируйте в описание PR вывод исполнения вашей программы **на CI** (через редактирование описания PR или комментарием, главное используйте пожалуйста спойлер для компактности)

**Мягкий дедлайн**: лекция 31 марта.

**Жесткий дедлайн**: вечер 7 апреля.

Задание 4.1.
=========

Потренируйтесь в использовании Ceres Solver - выполните все TODO в tests/test_ceres_solver.cpp

Задание 4.2.
=========

Добавьте учет радиальных дисторсий в src/phg/core/calibration.cpp

Задание 4.3.
=========

Выполните все TODO в tests/test_sfm_ba.cpp

В MeshLab можно не только смотреть на отдельные облака точек, но и сравнивать несколько:

1) Запустите MeshLab
2) Выделите несколько .ply файлов и drag&drop-ните их на MeshLab
3) Затем нажмите на иконку Align - A в кружке (справа нажимая на глазики можно скрыть или показать конкретное облако, и нажав галку показать цвета вершин или ложные цвета):

![MeshLab](/.github/screens/meshlab.png?raw=true)

P.S. если у вас случается ошибка ```unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.``` - вероятнее всего это обычный segfault (например выход за пределы массива), но почему то связка CLion + MSVC/Win это не обрабатывают корректно и не дают возможности увидеть строчку падения даже под отладчиком. Рекомендуется использовать Linux.

Задание 4.4.
=========

Приложите скриншоты своих лучших результатов. Хотя бы по saharov и herzjesu. Примеры:

![saharov32](/.github/screens/saharov32.png?raw=true)

![herzjesu25](/.github/screens/herzjesu25.png?raw=true)
