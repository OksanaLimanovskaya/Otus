# Система прогнозирования биовозраста по функциональным данным пациетна
## Цель модели: предсказание биовозраста пациента по набору функциональных данных. 
## Данные: 
База данных медицинской организации ГАУЗ СО «СОКП Госпиталь для ветеранов войн» и ГАУЗ СО
«Институт медицинских клеточных технологий» (г. Екатеринбург) за 1995-2022 гг. Из полученных данных были выделены функциональные показатели пациентов (10 показателей), впервые пришедших на обследование. Была проведена предварительная обработка информации, заключающаяся в удалении пропущенных значений. Итоговая выборка содержала 10 числовых признаков и 1185 записей.
Анализировался 10 показателей из предоставленного набора функциональных параметров:
1. АДС – артериальное давление систолическое в мм.рт.ст.,
2. АДД – артериальное давление диастолическое в мм.рт.ст.,
3. АДП – разность между систолическим и диастолическим давлением в мм.рт.ст.,
4. ЗДВдох – задержка дыхания на вдохе в секундах,
5. ЗДВыдох – задержка дыхания на выдохе в секундах,
6. ЖЕЛ – жизненная емкость легких в мл,
7. масса – масса тела в кг,
8. аккомодация в диоптриях,
9. острота слуха в бел,
10. статическая балансировка в секундах.

Кластеризация данных дала 4 кластера по полу и типу пациента - амбулаторный и стационарный. 
![cluster_res](https://github.com/OksanaLimanovskaya/Otus/assets/135599630/9f5bf323-4a27-432e-8944-f13ec1eab2e8)

[feature_importance.pdf](https://github.com/OksanaLimanovskaya/Otus/files/12840521/feature_importance.pdf)

[LimanovskayaSoavtors_2_23_1.pdf](https://github.com/OksanaLimanovskaya/Otus/files/12840523/LimanovskayaSoavtors_2_23_1.pdf)



Поэтому далее нужно строить модели для мужчин и женщин отдельно. Причем для уменьшения ошибки модели и отделения ее от дельты биовозраста решено использовать только данные для амбулаторных пациентов как практически здоровых.В итоге для построения модели для мужчин использовалась бд в объеме 344 записи, а для женщин в объеме 991.
## Визуальный анализ данных
Начнем с распределения целевой фичи - возраста
Для мужчин получилось
![age_men](https://github.com/OksanaLimanovskaya/Otus/assets/135599630/a51a2611-a7cf-4a61-bcf3-64a1553d57f5)
Для женщин
![age_women](https://github.com/OksanaLimanovskaya/Otus/assets/135599630/b052cef6-761e-48d1-8f8e-286f48ae3f60)


