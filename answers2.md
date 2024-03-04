1) Зачем фильтровать матчи, если потом мы запускаем устойчивый к выбросам RANSAC и отфильтровываем шумные сопоставления?
> Потому что без фильтрации выбросов крайне много. Настолько много, что RANSAC-у было бы сложно искать настоящие сопоставления, не похожие на шум.
> Также стоит вопрос быстродействия -- чем меньше матчей мы передаем в RANSAC, тем быстрее он будет выполнять голосование и тем больше его итераций мы можем запустить, сущетсвенно увеличивая точность.

2) Cluster filtering довольно хорошо работает и без Ratio test. Однако, если оставить только Cluster filtering, некоторые тесты начнут падать. Почему так происходит? В каких случаях наоборот, не хватает Ratio test и необходима дополнительная фильтрация?
> Без предварительной фильтрации по разумным сопоставлениям cluster filtering слишком много фильтрует из-за того, что многие ближайшие соседи перешли во что-то неразумное. Это легко увидеть по картинкам -- cluster filtering оставляет банально меньше точек, если его запускать не на уже отфильтрованных по k ratio test матчам.
> Самого по себе ratio теста нигде не хватает - это всего проверка лишь обрезает совсем подозрительные матчи, где есть две точки с очень похожими дескрипторами

3) С какой проблемой можно столкнуться при приравнивании единице элемента H33 матрицы гомографии? Как ее решить?
> Если параметризовать матрицу гомографии 9 неизвестными (т.е. всей матрицей 3x3), то четырьмя парами точка - ее образ у нас создается 8 уравнений-ограничений на них. Как известно, у нас получится пространство решений размерности >= 1 (9 неизвестных - 8 решений), причем одно из измерений из соображений проблематики - это домножение всей матрицы на константу, которое не меняет образы точек. Однако при этом H33 в этиъ домножениях всегда ненулевой или всегда нулевой. Если он всегда ненулевой, то его можно приравнять к единице и получить система, у которой, как правило, есть решение (мы такое и делали). Если же он всегда ноль, то наш метод не сработает. Однако само условие, что одна из переменных всегда равна нулю, не мешает решать нам систему в общем случае (можно решать систему любым другим способом), нужно только отказаться от предположения, что мы один из элементов делаем равным единице. После решения мы получим матрицу гомографии не в стандартном виде, но это все еще будет корректная матрица гомографии.

4) Какой подвох таится в попытке склеивать большие панорамы и ортофото методом, реализованным в данной домашке? (Для интуиции можно посмотреть на результат склейки, когда за корень взята какая-нибудь другая картинка)
> У нас будет накапливаться неточность -- если дерево получится глубоким, то неточность будет очень резко расти ввиду большого количества перемноженных неточных матриц.
> {-1, 0, 0, 2, 2} - поменял дерево на такое, стало ощутимо лучше
> 
> Я поподбирал еще деревья, ярко выраженного подвоха не нашел

5) Как можно автоматически построить граф для построения панорамы, чтобы на вход метод принимал только список картинок?
> Можно построить бамбук 😉. Более хорошего варианта нет -- деревьев, согласно теории кодов Прюфера, порядка n^n, а перестановок
> n!; на общее дерево заведомо не хватит информации в перестановке.

6) Если с вашей реализацией SIFT пройти тесты не получилось, напишите (если пробовали дебажить), где, как вам кажется, проблема и как вы пробовали ее решать.
> Даже имея идеальный sift из библиотеки и flann из библиотеки все еще нужно было очень филигранно уточнять параметры, чтобы тесты прошли. Моя реализация sift-а очень сырая, видно, что там на самом деле мало точек, которые соответствуют одним и тем же точкам и имеют схожие дескрипторы. Нет никаких шансов, что за конечное время у меня получится дотолкать ногами мой sift, чтобы он прошел тесты. Можно было бы снизить амбициозность и добиться хотя бы адекватных результатов; у меня на это не осталось времени, но если бы я хотел это сделать, я бы начал с написания более жестких тестов для SIFT-а, которые более аккуратно отслеживают соответствие между точками. 

7) Если есть, фидбек по заданию: какая часть больше всего понравилась, где-то слишком сложно/просто (что именно), где-то слишком мало ссылок и тд.
> Увидеть то, что написанный тобой алгоритм действительно хорошо склеивает фоточки в панораму - это 🔥
> В целом задания понравились. Было бы круто, если бы у функций, в которых что-то пропущено, вначале были краткие описания того, что они делают, чтобы можно было стереть функцию целиком и восстанавливать алгоритм по описанию (это, возможно, исключительно моя прихоть)