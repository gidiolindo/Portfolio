#Projeto Modelagem Estatística - Análise de Churn

'''
Estudo de Caso: Uma empresa busca verificar quais os principais fatores que levam os clientes a cancelar o serviço.
Objetivo do Projeto: Identificar Fatores de cancelamento e quantificar o impacto  de cada fator no churn.

'''

#Importação de pacotes
import pandas
import statsmodels.api
import seaborn
import matplotlib.pyplot
import plotly.express


#Configurar visualizaçõ de dados

pandas.set_option('display.float.format', lambda x: '%.4f' % x) 

#Importação de arquivo
nome_arquivo = 'C:\\Users\\giova\\OneDrive\\Documentos\\projetosGit\\Modelagem Estatistica\\dados.csv'
dados = pandas.read_csv(nome_arquivo)

#=============================
#Análise Exploratória de Dados
#=============================

print(dados.info())
print(dados.describe())
print(dados.describe(include='object'))

#1. Proporção do Churn
#Sim para cancelamento

cont_churn = dados['Churn'].value_counts().rename(index= {1: 'Sim', 0: 'Não'})
cores = ['#AEC7E8', '#2A9D8F']
print(cont_churn)

matplotlib.pyplot.pie(
    cont_churn,
    labels= cont_churn.index,
    autopct= '%1.2f%%',
    startangle= 180,
    colors= cores
)

matplotlib.pyplot.title('Taxa de Churn Geral')
matplotlib.pyplot.show()

numerador = cont_churn.get('Sim')
taxa = numerador/2000 
taxa = taxa * 100
print(f'A taxa de Churn da nossa base de clientes é {taxa:.2f}%')

# -> Através do gráfico de proporção verifica se uma alta taxa de churn


#2. Análise do Churn por tipo de contrato 

seaborn.countplot(
    data = dados,
    x = 'Tipo_Contrato',
    hue = 'Churn',
    palette= cores
)

matplotlib.pyplot.title('Taxa de Churn por Tipo de Contrato')
matplotlib.pyplot.xlabel('Tipo de Contrato')
matplotlib.pyplot.xticks(rotation = 0)
matplotlib.pyplot.ylabel('Número de Clientes')
matplotlib.pyplot.legend(title = 'Churn (0 = Não, 1 = Sim)')
matplotlib.pyplot.show()

#-> A partir da análise por tipo de contrato percebe se um alto cancelamento em contratos mensais  


#3. Fatura Mensal

histograma_fatura = plotly.express.histogram(
    dados,
    x = 'Fatura_Mensal',
    color = 'Churn',
    marginal = 'box',
    title = 'Distribuição da Fatura Mensal por Churn',
    labels= {'Fatura_Mensal' : 'Valor da Fatura Mensal'}
    )

histograma_fatura.show()

#-> Clientes com faturas mais altas de serviço possuem maior taxa de churn

#=============================
#Modelagem estatística
#=============================
 
#1. Conversão de variáveis (Tipo_Contrato e Servico_Internet)

modelo_estat = pandas.get_dummies(dados, columns= ['Tipo_Contrato', 'Servico_Internet'], drop_first= True, dtype=int)

print(modelo_estat.head())

#2. Definição de variáveis

y = modelo_estat['Churn']
X = modelo_estat.drop(['ID_Cliente', 'Churn'], axis = 1) 
X = statsmodels.api.add_constant(X)
print(X.head())

#3. Criação do modelo

modelo_logistico = statsmodels.api.Logit(y, X) #cria modelo de regressão logistica
modelo_treinado = modelo_logistico.fit() #treina modelo para entender relação entre variaveis

print(modelo_treinado.summary())

#=========================
#4. Resultado do Modelo
'''
coef (se positivo aumenta a chance de Churn): Tipo de Contrato Mensal e o Serviço de Fibra Óptica impactam na taxa de churn, pelo valor de p assume se que não é coincidência
'''
#=========================
'''
A partir da análise, recomenda se:
-> Converter clientes de contratos mensais para anuais
-> Oferecer pacotes de fatura vantajosos para conversão desses clientes, visto que o valor da fatura impacta no churn
-> Verificar o feedback dos clientes em relação ao serviço de Fibra Óptica, buscando entender quais aspectos do serviço oferecido atualmente levam ao cancelamento

'''