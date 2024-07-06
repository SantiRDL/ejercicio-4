import pandas as pd #para leer los datos de titanik.csv
import matplotlib.pyplot as plt #para graficar
import seaborn as sns #para graficar
import scipy.stats as stats #para hacer pruebas estadísticas

# Cargar los datos
dataframe = pd.read_csv('titanik.csv') #si la tabla está guardada en otro lugar, se debe cambiar la ruta o mover el archivo a la carpeta del proyecto.


# Calcular la media de las edades por género
mean_age_by_gender = dataframe.groupby('gender')['age'].transform('mean')

# Rellenar los valores faltantes
dataframe['age'] = dataframe['age'].fillna(mean_age_by_gender)

#Usar la media de las edades según el género es razonable porque las características de edad pueden variar entre géneros y esto proporciona una estimación más precisa.

#print(dataframe.head(10))


# Calcular estadísticas descriptivas
mean_age = dataframe['age'].mean()
median_age = dataframe['age'].median()
mode_age = dataframe['age'].mode()[0]
range_age = dataframe['age'].max() - dataframe['age'].min()
var_age = dataframe['age'].var()
std_age = dataframe['age'].std()

print(f"Media edad: {mean_age}")
print(f"Mediana edad: {median_age}")
print(f"Moda edad: {mode_age}")
print(f"Rango edad: {range_age}")
print(f"Varianza edad: {var_age}")
print(f"Desviación estándar edad: {std_age}")

# Tasa de supervivencia general
survival_rate = dataframe['survived'].mean()
print(f"Tasa de supervivencia general: {survival_rate * 100:.2f}%") #el :.2f se usa para cortar el número a dos decimales.

# Tasa de supervivencia por género
survival_rate_by_gender = dataframe.groupby('gender')['survived'].mean()
print(survival_rate_by_gender)

# Histograma de las edades por clase
sns.histplot(data=dataframe, x='age', hue='p_class', multiple='stack', kde=True)
plt.xlabel('Edad')
plt.ylabel('Cantidad de pasajeros')
plt.title('Histograma de las edades por clase')
plt.show()

# Diagrama de cajas para las edades de los supervivientes y no supervivientes
sns.boxplot(x='survived', y='age', data=dataframe)
plt.xlabel('Supervivencia (0 = No, 1 = Sí)')
plt.ylabel('Edad')
plt.title('Diagrama de cajas de las edades por estado de supervivencia')
plt.show()

# Intervalo de confianza del 95% para la edad promedio
conf_interval = stats.t.interval(0.95, len(dataframe['age'])-1, loc=mean_age, scale=std_age/len(dataframe['age'])**0.5)
print(f"Intervalo de confianza del 95% para la edad promedio: {conf_interval}")

# Prueba T para el promedio de edad de las mujeres
women_age = dataframe[dataframe['gender'] == 'female']['age']
t_stat, p_value = stats.ttest_1samp(women_age, 56)
print(f"Prueba T para mujeres: t={t_stat}, p={p_value}")

# Prueba T para el promedio de edad de los hombres
men_age = dataframe[dataframe['gender'] == 'male']['age']
t_stat, p_value = stats.ttest_1samp(men_age, 56)
print(f"Prueba T para hombres: t={t_stat}, p={p_value}")

# Prueba T para la tasa de supervivencia entre hombres y mujeres
survival_by_gender = dataframe.groupby('gender')['survived'].apply(list)
t_stat, p_value = stats.ttest_ind(survival_by_gender['male'], survival_by_gender['female'])
print(f"Diferencia en la tasa de supervivencia por género: t={t_stat}, p={p_value}")

# Prueba T para la tasa de supervivencia entre clases
survival_by_class = dataframe.groupby('p_class')['survived'].apply(list)
t_stat_1_2, p_value_1_2 = stats.ttest_ind(survival_by_class[1], survival_by_class[2])
t_stat_1_3, p_value_1_3 = stats.ttest_ind(survival_by_class[1], survival_by_class[3])
t_stat_2_3, p_value_2_3 = stats.ttest_ind(survival_by_class[2], survival_by_class[3])
print(f"Diferencia en la tasa de supervivencia entre primera y segunda clase: t={t_stat_1_2}, p={p_value_1_2}")
print(f"Diferencia en la tasa de supervivencia entre primera y tercera clase: t={t_stat_1_3}, p={p_value_1_3}")
print(f"Diferencia en la tasa de supervivencia entre segunda y tercera clase: t={t_stat_2_3}, p={p_value_2_3}")

# Prueba T para comparar la edad promedio entre hombres y mujeres
t_stat, p_value = stats.ttest_ind(women_age, men_age)
print(f"Diferencia en la edad promedio entre hombres y mujeres: t={t_stat}, p={p_value}")