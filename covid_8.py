# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import numpy as np
import scipy.optimize as sp
import os


'''
Analisis datos covid-19 en España 
4 de abril de 2020
Simple regresion asumiendo un comportamiento tipo sigmoide
Tomás E. Gómez Álvarez-Arenas
t.gomez@csic.es
www.us-biomat.com
'''

'''
ENTRADA DE DATOS

simplemente dos listas por cada dato: número y día
Se contemplan tres datos: casos / muertes / uci / altas
El ulitmo día añadido: 42, se refiere al 4 de abril
Se pueden añadir datos y días al cominezo de la lista a mano

O sustituir estas listas por datos de otro pais, CC.AA. etc...

'''
def intro_datos(quito=0):
    '''
    Aqui es donde introducimos los datos
    Para una mayor robustez, se emplean cuatro tipo de datos disponibles por el
    Ministerio de Sanidad, casos estudiados:
            Casos
            Muertos
            Casos en UCI
            Altas
    Como el criterio y la metodología es diferente para el registro de estos 
    datos cabe esperarse resultados contradictorios, pero el análisis de todos
    los datos de forma conjunta da más robustez al análisis
    
    Se crea un diccionario con cada uno de estos casos estudiados
    A continuación introduzco los datos como una lista con:
        Casos, días; en orden inverso

    Esta función devuelve un diccionario con una entrada para cada uno de los
    casos estudiados
    Para cada entrada proporciona tres items:
            Una lista de numero de casos
            Una lista con numero de días
            Una lista con dos np.arrays que contienen límites para el fitting
    
    El parámetro quito permite quitar días del estudio.
    Especifica el numero de días que se quitan empezando por el final.
    Sirve para chequear las predicciones con los datos diponibles en
    fechas anteriores
    quito = 0: no quito ningún día
    '''
    tipo_casos = ['casos', 'muertos', 'uci', 'altas']
    data = {'casos', 'muertos', 'uci', 'altas'}
    data = {}

    
    for tipo in tipo_casos:
        
        '''La variable PARAMS0 define unos límites para encontrar el mejor 
        ajuste de la función logistica a los datos. En ocasiones no es necesario
        (el fitting converge sin bounds, pero en otros casos puede ser 
        necesario)
        '''
            
        if tipo == 'casos':  
            PARAMS0 = (np.array([0.01e6, 0.01, 20]), np.array([1e6, 0.3, 70])) 
            # 28 de marzo
        if tipo == 'muertos':
            PARAMS0 = (np.array([0.01e4, 0.1, 30]), np.array([4e4, 0.5, 60]))   
            # 28 de marzo
        if tipo == 'uci': 
            PARAMS0 = (np.array([0.1e4, 0.1, 10]), np.array([6e4, 0.5, 70]))   
            # 28 de marzo
        if tipo == 'altas':
            PARAMS0 = (np.array([0.1e4, 0.1, 38]), np.array([0.1e6, 0.5, 70]))  
            # 28 de marzo


        '''
        Aqui introducimos los datos
        Se podrían leer de una tabla (excel o similar)... pero aqui los tengo 
        metidos directmente
        '''
        if tipo == 'casos':
            casos =[135032, 130759, 124736, \
            117710, 110238, 102136, 94417, 85195,  78797, 72248, 64059,         \
            56188, 47610., 39673, 33089,   28572, 24926, 19980, 17147, 14700,   \
            11178, 10000.0, 9191, 7753, 6315, 5147, 3000, 2222, 1622, 1200, 674,\
            441]
            # dia 42: 4 de abril
            dias = [44, 43, 42, \
            41,  40,    39,    38,    37,    36,   35,   34,            \
            33,    32,   31,    30,    29,    28,    27,   26,   25,            \
            24,    23,   22,    21,    20,    19,    18,   17,   16,   15,  14, \
            13]
            casos = casos[quito:]
            dias = dias[quito:]
            t0 = dias[-1]
            PARAMS0 = (np.array([0.01e6, 0.01, 20]), np.array([1e6, 0.3, 70]))  
            # 28 de marzo
            
        if tipo == 'muertos':
            casos = [13055, 12418, 11744, \
            10953, 10003,  9053, 8189, 7340, 6528, 5690, 4858,  4098,           \
            3435., 2696.0, 2182., 1720., 1326., 1002., 767.0, 636., 491.,       \
            390.00, 329.0, 288,  193., 120.]
            dias = [44, 43, 42, \
            41,    40,    39,   38,     37,    36,    35,   34,  33,            \
            32,     31,    30,    29,   28,     27,    26,    25,   24,         \
            23,     22,    21,    20,   19]
            casos = casos[quito:]
            dias = dias[quito:]
            t0 = dias[-1]
            PARAMS0 = (np.array([0.01e4, 0.1, 30]), np.array([3e4, 0.5, 60]))   
            # 28 de marzo
   
        if tipo == 'uci':
            casos =[6931, 6861, 6532, \
            6416, 6092, 5872, 5607, 5231, 4907,  4575, 4165, 3679,              \
            3166,  2636, 2355, 1785, 1612,  1141,  939,  774,  563,   432, ]
            dias = [44, 43, 42, \
            41,   40,   39,    38,    37,   36,   35,   34,    33,              \
            32,     31,   30,   29,    28,    27,   26,   25,   24,    23]
            casos = casos[quito:]
            dias = dias[quito:]
            t0 = dias[-1]
            PARAMS0 = (np.array([0.1e4, 0.1, 10]), np.array([6e4, 0.5, 70]))
    
        if tipo == 'altas':
            casos = [40437, 38080, 34219, \
            30513, 26743, 22647, 19259, 16780, 14709, 12285, 9357, 7015,        \
            5367, 3794,   3355,  2575,  2125, 1585, 1107, 1081, 1098, 530, 517]
            dias = [44, 43, 42, \
            41,    40,    39,    38,    37,   36,   35,    34,  33,             \
            32,     31,    30,    29,    28,    27,   26,   25,    24,  23,  22]
            casos = casos[quito:]
            dias = dias[quito:]
            PARAMS0 = (np.array([0.1e4, 0.1, 38]), np.array([0.1e6, 0.5, 70]))
            PARAMS0 = (np.array([0.13e6, 0.1, 38]), np.array([0.19e6, 0.5, 70]))   
            # 31 de marzo
            
        dias = dias[::-1]
        casos = casos[::-1]
        data[tipo] =[casos, dias, PARAMS0]
    return data



'''
Primero definimos algunas funciones para el calculo
'''

def write_data_to_csv(FF0, resultados, dicc_dias, QUITO=0):
    Lista_head = ['Dia de prediccion',	'Casos totales (mil)',                  \
                  'Pico de casos x dia', 'Fallecimientos totales (mil)',        \
                  'Pico de fallecimientos por dia', 'Pico de altas por dia',    \
                  'Pico de casos activos',                                      \
                  'Superar 95% casos, 99% defunciones, 95% altas']

        
    if os.path.exists(FF0):
        print 'Extacted data file already exists, data added at the end'
        FF0 = open(FF0, 'a')
    else:
        FF0 = open(FF0, 'a')
        print 'Extacted data file and header created'
        for n in Lista_head: 
            FF0.write( n + '; ' + ' ')
        FF0.write('\n')
        
    uu = intro_datos(quito=QUITO)
    dia_pred_no =  uu['casos'][1][-1]
    dia_pred = dicc_dias[dia_pred_no]
    FF0.write(dia_pred + ';')
    FF0.write(str(int(resultados[0][0][0] / 1e3)) + ' - '                       \
                + str(int(resultados[1][0][0] / 1e3)) + ';') 
    FF0.write(str(dicc_dias[int(resultados[0][0][2])]) + ' - '                  \
                +  str(dicc_dias[int(resultados[1][0][2])]) + ';')
    FF0.write(str(int(resultados[2][0][0] / 1e3)) + ' - '                       \
                + str(int(resultados[3][0][0] / 1e3)) + ';')
    FF0.write(str(dicc_dias[int(resultados[2][0][2])]) + ' - '                  \
                + str(dicc_dias[int(resultados[3][0][2])]) + ';')
    FF0.write(str(dicc_dias[int(resultados[4][0][2])]) + ' - '                  \
                + str(dicc_dias[int(resultados[5][0][2])]) + ';')
    FF0.write( str(dicc_dias[int(resultados[8])])                               \
                 + ' - ' + str(dicc_dias[int(resultados[9])]) + ';')
    FF0.write(str(dicc_dias[int(resultados[10])]) + ' - '                       \
             + str(dicc_dias[int(resultados[11])]) +                            \
             ' - ' + str(dicc_dias[int(resultados[12])]) + ';')
    FF0.write('\n')
    FF0.close()
    return       

def days_dict():
    '''
    Esto crea un diccionario para mapear la variable no. de dias en 
    dias de calendario
    '''
    dd = {}
    diak = 1
    for n in range(7):
        dia = 23 + n
        diad = str(dia) + ' de Febrero'
        dd[diak] = diad
        diak += 1
    for n in range(31):
        dia = 1 + n
        diad = str(dia) + ' de Marzo'
        dd[diak] = diad
        diak += 1
    for n in range(30):
        dia = 1 + n
        diad = str(dia) + ' de Abril'
        dd[diak] = diad
        diak += 1
    for n in range(31):
        dia = 1 + n
        diad = str(dia) + ' de Mayo'
        dd[diak] = diad
        diak += 1
    return dd       

def print_results(dicc_dias, resultados):
    '''
    Esta funcion imprime por pantalla algunos de los resultados
    que se obtienen del análisis junto con su definición
    '''
    print '  '
    print 'Maximo numero de casos: ' + str(int(resultados[0][0][0])) + ' - '   \
                + str(int(resultados[1][0][0]))
    print 'Día pto inflexion (casos por dia): '                                \
                + str(dicc_dias[int(resultados[0][0][2])]) + ' - '              \
                +  str(dicc_dias[int(resultados[1][0][2])])
    print ' '
    print 'Maximo numero de muertos: ' + str(int(resultados[2][0][0])) + ' - ' \
                + str(int(resultados[3][0][0]))
    print 'Día pto inflexion (muertos por dia): '                              \
                + str(dicc_dias[int(resultados[2][0][2])]) + ' - '              \
                + str(dicc_dias[int(resultados[3][0][2])])
    print ' '
    
    print 'Día maximo altas por dia: '                                         \
                + str(dicc_dias[int(resultados[4][0][2])]) + ' - '              \
                + str(dicc_dias[int(resultados[5][0][2])])
    print ' '
    
    print 'Día pico casos activos: ' + str(dicc_dias[int(resultados[8])])      \
                 + ' - ' + str(dicc_dias[int(resultados[9])])
       
    print ' '
    normmm = np.max([int(resultados[10]), int(resultados[11]),                  \
                     int(resultados[12]), int(resultados[13])])
    print 'Superación 95% de altas totales: ' + str(dicc_dias[normmm])
    print ' '
    print ' '
    print 'Superar 95% de casos / 99% fallecimientos): '                       \
             + str(dicc_dias[int(resultados[10])]) + ' - '                      \
             + str(dicc_dias[int(resultados[11])])
    print 'Superar el 95% de altas / ingresados uci): '                        \
             + str(dicc_dias[int(resultados[12])])                              \
             + ' - ' + str(dicc_dias[int(resultados[13])])
     
    return
    

def logistic(x, MAX, k, x0):
    '''Esta es la funcion logistica que se emplea para describir muchos procesos
    en física, biología, etc...
    Parece que es una ley que podría ajustarse bien a fenómenos como la difusión
    del covid19
    La función toma tres párametros:
         MAX: total de casos (float)
         k: velocidad de transición de la curva (float)
         x0: ubicación del pto. de inflesión  (float)
    Y la variable días: x (np.arryay or list 1D)
    
    Devuelve y (np.array 1D): número de casos (acumulado) por día
    '''
    y = MAX / (1 + np.exp(-k * (x - x0) ))
    return y

def casos_x_dia(x, MAX, k, x0):
    '''Esta función calcula, a partir de lla función logística anterior
    el numero de casos x día
    Toma los mismo argumentos que la función logistica
    '''
    y0 = MAX / (1 + np.exp(-k * (x - x0) ))
    y1 = MAX / (1 + np.exp(-k * (x + 1 - x0) ))
    y = y1 - y0
    return y
    
def Rcasos_acumulados(casos_dia):
    ''' esta función calcula el numero de casos acumulados apartir del
    dato de número de casos por día
    Toma una lista (o np.array 1D) con el número de casos por día
    Devuelve una lista con el número de casos acumulado
    '''
    casos_a = []
    ind = 0
    for nn in casos_dia:
        if ind == 0:
            casos_a.append(casos_dia[0])
        else:
            casos_a.append(sum(casos_dia[0:ind]))
        ind += 1
    return casos_a
    
def Rcasos_x_dia(casos):
    '''Esta función calcula el número de casos por día a partir del 
    número de casos acumulados
    Toma una lista (o np.array 1D) con el número de casos acumulado
    Devuelve una lista con el número de casos por día
    '''
    casos_xd = []
    ind = 0
    for nn in casos:
        if ind == 0:
            casos_xd.append(casos[0])
        else:
            casos_xd.append(casos[ind] - casos[ind-1])
        ind += 1 
    return casos_xd

def f_analisis(data,T0=5, T1=10, MortMax=10., TUCI0=-2, TUCI1=5, TALTAS0=15,    \
                TALTAS1=20, plot_display=True):
    '''
    Esta función lleva a cabo el análisis de los datos
    esto consiste en ajustar con una función logistica los cuatro tipo de datos
    analizadso:
            casos
            muertos
            altas
            uci
   
   Toma la variable data construida por la funcion: intro_datos()
   Además toma parametros:
      T0, T1, dos numeros enteros, T1 > T0
         Cuantifican el límite superior en inferior del retraso en días que se
         espera para la sigmoide que representa el número de defunciones con 
         respecto a la que representa el número de casos detectados
         Los valores por defecto son: entre 5 y 10 días
     MortMax: float, representa la mortalidad máxima esperada 
               (en %: 0.0 - 100.0%)
     TUCi0, TUCI1, dos numeros enteros TUCI1 > TUCI0
         Cuantifican el límite superior en inferior del retraso en días que se
         espera para la sigmoide que representa el número de ingresos UIC con 
         respecto a la que representa el número de casos detectados.
         Curiosamente los datos del ministerio parecen indicar que la sigmoide
         de ingresos UCI esta adelantada con respecto a la de casos. Por esto
         los limites por defecto son entre -2 y 4 días
     TALTAS0, TALTAS1: dos numeros enteros, TALTAS1 > TALTAS0
         Cuantifican el límite superior en inferior del retraso en días que se
         espera para la sigmoide que representa el número de altas con 
         respecto a la que representa el número de casos detectados
         Los valores por defecto son: entre 15 y 20 días 
     plot_display: variable lógica: True or False, permite activar o desactivar
         la generación de gráficas
   
   Además, para cada caso, se analizan dos formas de ver el problema:
        Ajustando casos acumulados vs dias con una función logistica
        Ajustando casos x día con la función casos_x_dia introducida antes
    
    Esto permite un mayor grado de robustez y también determinar rangos
    de incertidumbre para la predicción
    
    Finalmente se calcula una curva de casos activos vs días
    que es el resultado de restar al numero de casos estimado el número de 
    muertos estimado y el número de altas estimado
    
    Igual que antes, esto puede hacerse empleando directamente los datos de 
    casos acumulados o los datos de casos acumulados calculados a partir de los 
    casos por día.
    Cabe esperar un mayor error en esta segunda prediccón, aunque si todo fuera
    ideal ambos resultados deberían coincidir
    
    Como es normal, los errores se progapan y esta curva puede presentar 
    características irreales, pero también da una estimación de la ubucación 
    del pico de la pandemia entendido como pico el pico de esta curva
    
    La estimación de vuelta a la normalidad puede hacerse estimando que este día
    será el día en el que el número de casos totales sea >90% del número total
    de casos que la curva logistica preve.
    '''
    itera = 1
    MAX_CASOS = 1
    tipo_casos = ['casos', 'muertos', 'uci', 'altas']
    for nn in tipo_casos:
        casos = data[nn][0]
        dias = data[nn][1]
        PARAMS0 = data[nn][2]
        if nn == 'altas':
            ''' Se impone como limite inferior que el numero de altas sea, al
            menos, igual a la mortalidad máxima MortMax, por defecto: 10%
            ''' 
            FACT = (100. - MortMax) / 100.
            PARAMS0 = (np.array([MAX_CASOS * FACT, PARAMS0[0][1],               \
                                int(XXXc[0][2]) + TALTAS0 ] ),                  \
                       np.array([MAX_CASOS, PARAMS0[1][1],                      \
                                  int(XXXc[0][2]) + TALTAS1 ] ) )  
                       
        if nn == 'muertos':
            ''' Se impone que la sigmoide que refleja el numero de 
            fallecimientos lleva un retraso de entre T0 (5 por defecto) y T1 
            (10 por defecto días comoparada con la sigmoide que refleja el 
            número de casos detectados
            '''
            PARAMS0 = (np.array([PARAMS0[0][0], PARAMS0[0][1], \
                                               int(XXXc[0][2]) + T0 ]), \
                       np.array([PARAMS0[1][0], PARAMS0[1][1], \
                                               int(XXXc[0][2] + T1)])) 
        if nn == 'uci':
            ''' Se impone que la sigmoide que refleja el numero de 
            fallecimientos
            lleva un retraso de entre TUCI0 (5 por defecto) y TUCI1 (10 por 
            defecto días comoparada con la sigmoide que refleja el número de 
            casos detectados
            '''
            PARAMS0 = (np.array([PARAMS0[0][0], PARAMS0[0][1], \
                                               int(XXXc[0][2]) + TUCI0]), \
                       np.array([PARAMS0[1][0], PARAMS0[1][1], \
                                               int(XXXc[0][2] + TUCI1)]))    
        casos_dia = Rcasos_x_dia(casos)
        
        XXX = sp.curve_fit(logistic, dias, casos, bounds=PARAMS0) 
        YYY = sp.curve_fit(casos_x_dia, dias, casos_dia, bounds=PARAMS0)
        #print XXX
        #print YYY
    
        dias_t = np.linspace(1,70,100)
        
        ''' Casos totales '''
        casos1_t = logistic(dias_t, XXX[0][0], XXX[0][1], XXX[0][2])
        ''' Casos por día calculados a partir del ajuste de los casos totales'''
        casos_x_dia_acumulado = Rcasos_x_dia(casos1_t)
        
        ''' Casos x día'''
        casos_t = casos_x_dia(dias_t, YYY[0][0], YYY[0][1], YYY[0][2])
        ''' Casos totales calculados a pertir del ajuste de casos por día'''
        #casos_acumulado_x_dia = Rcasos_acumulados(casos_t)
        casos_acumulado_x_dia = \
                       logistic(dias_t, YYY[0][0], YYY[0][1], YYY[0][2])
              
        maximo_ = np.where(casos_t == max(casos_t))[0][0]
        maximo = dias_t[maximo_]
        
        label_='Pto. inflexion: dia ' + str(maximo )[0:2]
        label2_= 'Pico: dia ' + str(maximo )[0:2]
        
        maximo_ = np.where(casos_x_dia_acumulado == \
                            max(casos_x_dia_acumulado))[0][0]
        maximo = dias_t[maximo_]
        
        label_1='Pto. inflexion: dia ' + str(maximo )[0:2]
        label2_1= 'Pico: dia ' + str(maximo )[0:2]
             
        if plot_display:
            plt.figure(itera)
            plt.subplot(221)
            plt.plot(dias_t, casos_acumulado_x_dia, label= label_)
            plt.plot( dias_t, casos1_t, label= label_1)
            plt.plot(dias, casos,'o' )
            plt.legend()
            plt.xlabel('Dias')
            plt.title('Dia 1: 23 febrero, dia 39: 1 de abril, dia 69: 1 mayo')
            plt.ylabel('Numero de casos totales')
            plt.xlim(10,80)
            plt.grid()
            plt.show()
            
            plt.subplot(222)
            plt.semilogy(dias_t, casos_acumulado_x_dia, label= label_)
            plt.semilogy(dias_t, casos1_t, label = label_1)
            plt.plot(dias, casos,'o')
            plt.legend()
            plt.xlabel('Dias')
            plt.xlim(10,80)
            if nn == 'casos':
                plt.title('Casos = Casos detectados')
                plt.ylim(100, PARAMS0[1][0]*1.1)
                MAX_CASOS = XXX[0][0]
            if nn == 'muertos':
                plt.title('Casos = Defunciones')
                plt.ylim(10, PARAMS0[1][0]*1.1)
            if nn == 'uci':
                plt.title('Casos = Ingresados en UCI')
                plt.ylim(10, PARAMS0[1][0]*1.1)
            if nn == 'altas':
                plt.title('Casos = Altas')
                plt.ylim(10, PARAMS0[1][0]*1.1)
            plt.ylabel('Numero de casos totales')
            plt.grid()
            
            plt.subplot(223)
            plt.plot(dias_t, casos_t, label= label2_)
            plt.plot(dias_t, casos_x_dia_acumulado, label=label2_1)
            plt.plot(dias, casos_dia, 'o')
            plt.legend()
            plt.xlabel('Dias')
            plt.ylabel('Casos nuevos diarios')
            plt.grid()
            plt.xlim(10,80)
            
            plt.subplot(224)
            plt.semilogy(dias_t, casos_t, label=label2_)
            plt.semilogy(dias_t, casos_x_dia_acumulado, label=label2_1)
            plt.semilogy(dias, casos_dia, 'o')
            plt.xlabel('Dias')
            plt.legend()
            plt.ylabel('Casos nuevos diarios')
            if nn == 'casos':
                plt.ylim(100, np.max([np.max(casos_t), \
                        np.max(casos_x_dia_acumulado), np.max(casos_dia)])*2)
            if nn == 'muertos':
                plt.ylim(50, np.max([np.max(casos_t), \
                        np.max(casos_x_dia_acumulado), np.max(casos_dia)])*2)
            if nn == 'uci':
                plt.ylim(50, np.max([np.max(casos_t), \
                        np.max(casos_x_dia_acumulado), np.max(casos_dia)])*1.5)
            if nn == 'altas':
                plt.ylim(100, np.max([np.max(casos_t), \
                        np.max(casos_x_dia_acumulado), np.max(casos_dia)])*2)
            plt.grid()
            plt.xlim(10,80)
            plt.show()
            
        if nn == 'casos':
            casos_logistic = casos1_t

            casos_logistic2 = casos_acumulado_x_dia
            norm_cc = np.where(casos1_t > 0.95 * np.max(casos1_t))[0][0]
            norm_c =dias_t[norm_cc]
            XXXc = XXX
            YYYc = YYY
        if nn == 'altas':
            altas_logistic = casos1_t
            altas_logistic2 = casos_acumulado_x_dia
            norm_aa = np.where(casos1_t > 0.95 * np.max(casos1_t))[0][0]
            norm_a =dias_t[norm_aa]
            XXXa = XXX
            YYYa = YYY
        if nn == 'muertos':
            muertos_logistic = casos1_t
            muertos_logistic2 = casos_acumulado_x_dia
            norm_mm = np.where(casos1_t > 0.99 * np.max(casos1_t))[0][0]
            norm_m =dias_t[norm_mm]
            XXXm = XXX
            YYYm = YYY
        if nn == 'uci':
            uci_logistic = casos1_t
            uci_logistic2 = casos_acumulado_x_dia
            norm_uu = np.where(casos1_t > 0.95 * np.max(casos1_t))[0][0]
            norm_u =dias_t[norm_uu]
            XXXu = XXX
            YYYu = YYY            
        itera += 1
    
    Activos = casos_logistic - altas_logistic - muertos_logistic 
    Activos2 = np.array(casos_logistic2) - np.array(altas_logistic2) \
               - np.array(muertos_logistic2) 
    Pico = dias_t[(np.where(Activos == np.max(Activos))[0][0])]
    Pico2 = dias_t[(np.where(Activos2 == np.max(Activos2))[0][0])]
    if plot_display:
        label_ = 'Pico de la curva (datos acumulados): dia ' + str(Pico)[0:2]
        label2_ = 'Pico de la curva (datos x dia): dia ' + str(Pico2)[0:2]
        plt.figure(5)
        plt.plot(dias_t, Activos, label=label_)
        plt.plot(dias_t, Activos2, label=label2_)
        MAXX = max(np.max(Activos), np.max(Activos2))
        plt.ylim(0,MAXX * 1.2)
        plt.legend()
        plt.xlabel('Dias')
        plt.title('Dia 1: 23 febrero, dia 39: 1 de abril, dia 69: 1 mayo')
        plt.ylabel('Numero de casos activos')
        plt.grid()
        plt.show()
        plt.figure(6)
        DELAY = 0
        if DELAY == 0:
            plt.plot(dias_t[0:], muertos_logistic[DELAY:] / casos_logistic[0:], \
                  label=label_)
            plt.plot(dias_t[0:],                                                \
            np.array(muertos_logistic2[DELAY:]) / np.array(casos_logistic2[0:]),\
                label=label2_)
        if DELAY > 0:
            plt.plot(dias_t[0:-DELAY],                                          \
                    muertos_logistic[DELAY:] / casos_logistic[0:-DELAY],        \
                    label=label_)
            plt.plot(dias_t[0:-DELAY],                                          \
                        np.array(muertos_logistic2[DELAY:]) /                   \
                              np.array(casos_logistic2[0:-DELAY]),              \
                    label=label2_)

        #plt.legend()
        plt.xlabel('Dias')
        plt.title('Dia 1: 23 febrero, dia 39: 1 de abril, dia 69: 1 mayo')
        plt.ylabel('Mortalidad: Fallecimientos / Casos')
        plt.grid()
        plt.show()
    return XXXc, YYYc, XXXm, YYYm, XXXa, YYYa, XXXu, YYYu, Pico, Pico2, norm_c,\
            norm_m, norm_a, norm_u
        
def hacer_algo():  
    plt.close('all')
    aa = days_dict()
    data = intro_datos(quito=0)
    '''El parámetro quito permite quitar días del estudio.
    Especifica el numero de días que se quitan empezando por el final.
    Sirve para chequear las predicciones con los datos diponibles en
    fechas anteriores
    quito = 0: no quito ningún día
    '''
    FAR = f_analisis(data, T0=5, T1=10, MortMax=10., TUCI0=-2, TUCI1=5, \
                      TALTAS0=15, TALTAS1=20, plot_display=True)
    ''' No hago más que explicitar los valores por defecto de los parámetros
    que toma la función para simplificar la modificación que se quiera hacer
    Ver la info de la función para ver lo que significa cada uno de los 
    parámetros
    '''
    print_results(aa, FAR)
    write_data_to_csv('Tabla_predic2', FAR, aa, QUITO=0)
    
    
def hacer_algo3(QUITO=0, GRAF=True):  
    plt.close('all')
    aa = days_dict()
    data = intro_datos(quito=QUITO)
    '''El parámetro quito permite quitar días del estudio.
    Especifica el numero de días que se quitan empezando por el final.
    Sirve para chequear las predicciones con los datos diponibles en
    fechas anteriores
    QUITO = 0: no quito ningún día
    '''
    FAR = f_analisis(data, T0=5, T1=10, MortMax=10., TUCI0=-2, TUCI1=5, \
                      TALTAS0=15, TALTAS1=20, plot_display=GRAF)
    ''' No hago más que explicitar los valores por defecto de los parámetros
    que toma la función para simplificar la modificación que se quiera hacer
    Ver la info de la función para ver lo que significa cada uno de los 
    parámetros
    '''
    print_results(aa, FAR)
    write_data_to_csv('Tabla_predic2', FAR, aa, QUITO)   

if __name__ == "__main__": 
    '''
    Elegir entre hacer_algo() y hacer_algo2(N)
    Dejar una y comentar la otra
    '''  
    pp = hacer_algo() 
    '''
    Procesamos todos los datos y generamos gráfica
    '''
    #pp = hacer_algo3(1)
    '''
    Procesamos todos los datos quitando desde 0 hasta 10 días
    y visualizamos las predicciones que se hacian en días anteriores
    '''
    #pp = hacer_algo3(QUITO=7)
        
    