#Projeto Análise Exploratória de Dados de Vendas

'''
Estudo de Caso: Uma empresa de e-commerce possui um volume significativo de dados sobre suas vendas online, contudo por inconsistências nos dados não é possível gerar análises assertivas sobre as vendas
Objetivo do Projeto: Realizar o tratamento dos dados e análise exploratória
'''

#Importação de pacotes
import pandas
import numpy
import matplotlib.pyplot
from matplotlib.ticker import FuncFormatter
import seaborn

def formatamoeda(valor_eixo, posicao):
    '''
    Docstring for formatamoeda
    Formatação dos valores dos gráficos.
    
    :param valor_eixo: valor
    :param posicao: posição
    
    '''
    return f'{valor_eixo:,.0f}'.replace(',', '.')

#Padronização dos gráficos gerados
seaborn.set_style('whitegrid')
matplotlib.pyplot.rcParams['figure.figsize'] = (12,6)

#===================================
#Gerar base de dados fictícios 
#===================================
numpy.random.seed(42)

dados_vendas = {
    'ID_Pedido': numpy.random.randint(1000, 1101, size = 100),
    'Data_Compra': pandas.to_datetime(pandas.date_range(start = '2025-03-01', periods = 100, freq = 'D')) - pandas.to_timedelta(numpy.random.randint(0, 30, size = 100), unit = 'd'),
    'ID_Cliente': numpy.random.randint(100, 150, size = 100),
    'Produto': numpy.random.choice(['Smartphone', 'Notebook', 'Fone de ouvido', 'SmartWatch', 'Teclado Mecânico'], size = 100),
    'Quantidade': numpy.random.randint(1, 5, size = 100),
    'Preço_Unitario': [5999.90, 8500.00, 799.50, 2100.00, 850.00] *20,
    'Status_Entrega': numpy.random.choice(['Entregue', 'Pendente', 'Cancelado'], size = 100, p = [0.8, 0.15, 0.05])
}

DataFrame_vendas = pandas.DataFrame(dados_vendas)
DataFrame_vendas['Categoria'] = DataFrame_vendas['Produto'].apply(lambda x: 'Eletrônicos' if x in ['Smartphone', 'Notebook'] else 'Acessórios')


#Gerar Problemas de valores ausentes

DataFrame_vendas.loc[5:10, 'Quantidade'] = numpy.nan
DataFrame_vendas.loc[20:22, 'Status_Entrega'] = numpy.nan
DataFrame_vendas.loc[30, 'ID_Cliente'] = numpy.nan

#Gerar dados duplicados das 3 primeiras linhas
DataFrame_vendas = pandas.concat([DataFrame_vendas, DataFrame_vendas.head(3)], ignore_index= True) 

#Gerar tipos de dados incorretos
DataFrame_vendas['Preço_Unitario'] = DataFrame_vendas['Preço_Unitario'].astype(str)
DataFrame_vendas.loc[15,'Preço_Unitario'] = 'valor_invalido'
DataFrame_vendas['ID_Cliente'] = DataFrame_vendas['ID_Cliente'].astype(str)

#Gerar Outliers
DataFrame_vendas.loc[70, 'Quantidade'] = 50 

print('Dados gerados com sucesso!')

#===============================
#Análise Exploratória de Dados
#===============================

#Objetivo: identificar erros na base de dados (tipo de dados, valores ausentes, outliers etc)
print('-'*50)
print('Informações gerais da base:')
print('-'*50)
print(DataFrame_vendas.info())
print('-'*50)
print(f'Qnt de Valores ausentes:\n{DataFrame_vendas.isna().sum()} ')
print('-'*50)
print(f'Número de registros duplicados: {DataFrame_vendas.duplicated().sum()}')
print('-'*50)
print('Estatística Descritiva - Colunas Numéricas')
print(DataFrame_vendas.describe())
print('-'*50)
print('Estatística Descritiva - Colunas Categóricas')
print(DataFrame_vendas.describe(include= [object]))
print('-'*50)
print('Tipo de Dados')
print(DataFrame_vendas.dtypes)


#==============================
#LIMPEZA E PRÉ PROCESSAMENTO
#==============================

#Copia para evitar perda de dados
DataFrame_vendas_limpo = DataFrame_vendas

#1. Corrigir tipo de dados para ID_Cliente e Preco_Unitario

#coerce força a manter NAn
DataFrame_vendas_limpo['ID_Cliente'] = pandas.to_numeric(DataFrame_vendas_limpo['ID_Cliente'], errors = 'coerce').astype('Int64') 
DataFrame_vendas_limpo['Preço_Unitario'] = pandas.to_numeric(DataFrame_vendas_limpo['Preço_Unitario'], errors='coerce')

print(DataFrame_vendas_limpo.info())

#2. Tratar valores ausentes

#Para o tratamento da coluna Quantidade será usada a Mediana
medianaQtd = DataFrame_vendas_limpo['Quantidade'].median()
DataFrame_vendas_limpo.fillna({'Quantidade': medianaQtd}, inplace = True)

#Para o tratamento da coluna Status Entrega será usada Moda
modaStatus = DataFrame_vendas_limpo['Status_Entrega'].mode()[0] 
DataFrame_vendas_limpo.fillna({'Status_Entrega': modaStatus}, inplace = True)

#ID Cliente e Preço não é possível inferir -> será tratado através da exclusão

DataFrame_vendas_limpo = DataFrame_vendas_limpo.dropna(subset=['ID_Cliente','Preço_Unitario'])


#3. Remover duplicados

DataFrame_vendas_limpo.drop_duplicates(inplace=True)
print(DataFrame_vendas_limpo.info())

#4. Verificar outliers

seaborn.boxplot(x = DataFrame_vendas_limpo['Quantidade'])
matplotlib.pyplot.title('Boxplot de Quantidade')
matplotlib.pyplot.show()

#remover valores distantes 3 desvio padrão da media
limite_superior = DataFrame_vendas_limpo['Quantidade'].mean() + 3 * DataFrame_vendas_limpo['Quantidade'].std()
DataFrame_vendas_limpo = DataFrame_vendas_limpo[DataFrame_vendas_limpo['Quantidade'] < limite_superior]

#Resultado do tratamento de Outlier

seaborn.boxplot(x = DataFrame_vendas_limpo['Quantidade'])
matplotlib.pyplot.title('Boxplot de Quantidade')
matplotlib.pyplot.show()

#Validação após limpeza

print(DataFrame_vendas_limpo.info())
print('Valores ausentes: ', DataFrame_vendas_limpo.duplicated().sum())

#==========================
#Engenharia de Atributos
#==========================

#1. Cálculo do Total por pedido

DataFrame_vendas_limpo['Total'] = DataFrame_vendas_limpo['Quantidade']*DataFrame_vendas_limpo['Preço_Unitario']

#2. Total de receita de pedidos entregues (cancelados e pendentes não entram na receita)

entregues = DataFrame_vendas_limpo[DataFrame_vendas_limpo['Status_Entrega'] == 'Entregue']
print(entregues.head())
receita_vendas = entregues['Total'].sum()
print(f'A receita de vendas é: R$ {receita_vendas:.2f}')

#3. Receita por categoria

receita_categoria = entregues.groupby('Categoria')['Total'].sum().sort_values(ascending=False) #organizar decrescente
print('-'*30)
print('A receita por categoria é: ')
print(receita_categoria)

#4. Produto mais vendido em quantidade

produto_maisvendido = entregues.groupby('Produto')['Quantidade'].sum().sort_values(ascending=False)
print('-'*30)
print('A quantidade de produtos vendidos é:')
print(produto_maisvendido)

#Receita ao longo do tempo

receita_dia = entregues.set_index('Data_Compra').resample('D')['Total'].sum() #total de todos os dias
print('-'*30)
print('A quantidade de produtos vendidos é:')
print(receita_dia)


#===========================
#Visualização de dados
#===========================

#Gráfico por categoria

grafico = receita_categoria.plot(kind= 'bar', color = '#006a89')
grafico.yaxis.set_major_formatter(FuncFormatter(formatamoeda))
matplotlib.pyplot.title('Receita por Categoria')
matplotlib.pyplot.xlabel('Categoria')
matplotlib.pyplot.ylabel('Receita (R$)')
matplotlib.pyplot.xticks(rotation = 0)
matplotlib.pyplot.show()

#Gráfico quantidade vendida por produto
produto_maisvendido.plot(kind = 'barh', color = '#90b1db')
matplotlib.pyplot.title('Quantidade vendida por produto')
matplotlib.pyplot.xlabel('Quantidade (unidades)')
matplotlib.pyplot.ylabel('Produto')
matplotlib.pyplot.gca().invert_yaxis() 
matplotlib.pyplot.show()

#Gráfico tendência de vendas pelo tempo
linha = receita_dia.plot(kind = 'line', marker = '.', linestyle = '--')
linha.yaxis.set_major_formatter(FuncFormatter(formatamoeda))
matplotlib.pyplot.title('Receita por dia')
matplotlib.pyplot.xlabel('Data')
matplotlib.pyplot.ylabel('Receita (R$)')
matplotlib.pyplot.grid(True)
matplotlib.pyplot.show()

#Gráfico Distribuição Status Entrega

status_cont = DataFrame_vendas_limpo['Status_Entrega'].value_counts() 
maior_indice = status_cont.argmax()
explode = [0.1 if i == maior_indice else 0 for i in range(len(status_cont))]

matplotlib.pyplot.pie(status_cont,
                      labels = status_cont.index,
                      startangle= 180,
                      autopct= '%1.1f%%',
                      colors= ['#acc0c4','#cddbde','#e5eff0'],
                      explode=  explode,
                      shadow= True
)   
matplotlib.pyplot.title('Status de Entrega')
matplotlib.pyplot.show()
