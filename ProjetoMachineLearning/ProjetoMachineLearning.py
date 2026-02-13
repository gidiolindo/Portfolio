#Projeto Machine Learning - Análise de Sentimento

'''
Estudo de Caso: Uma empresa de e-commerce deseja automatizar a análise de feedback de seus clientes. 
Objetivo do Projeto: Criar um modelo de Machine Learning que classifique automaticamente os reviews de produtos como 'positivo' ou 'negativo'.
'''

#Importação de pacotes
import pandas
import re
import unicodedata
import seaborn
import matplotlib.pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#Carregamento de dados

nome_arquivo = 'dataset.csv'
data_base = pandas.read_csv(nome_arquivo)

#Análise exploratória dos dados
print(data_base.shape)
print(data_base.head())
print(data_base.tail())
print(data_base.sample())
print('-'*50)
print(f'\nResumo dos dados - objetos/Valores ausentes\n')
print(data_base.info())
print('-'*50)
print(f'\n Valores ausentes: \n')
print(data_base.isnull().sum())

'''
Coluna Review - Variável de entrada
Coluna Sentimento - Saída
'''

print('-'*50)
print(f'\nDistribuição dos sentimentos\n')
seaborn.countplot(data= data_base, 
                  x = 'sentimento',
                  palette=['#8ecae6', '#023047'])
matplotlib.pyplot.title('DISTRIBUIÇÃO DE SENTIMENTO')
matplotlib.pyplot.ylabel('Quantidade')
matplotlib.pyplot.xlabel('Sentimento')
matplotlib.pyplot.show()

#=======================
#Limpeza de dados
#=======================

#1. Remover linhas com valores ausentes
data_base.dropna(subset= ['texto_review'], inplace= True) 
print('-'*50)
print(f'\nDATA BASE ATUALIZADO\n')
print(data_base.info())

#2. Limpeza de texto

def limpa_texto(texto):
    """
    Função completa de limpeza de texto:
    1. Converte para minúsculas.
    2. Remove acentos e cedilha.
    3. Remove pontuações, números e caracteres especiais.
    4. Remove espaços extras.
    """
    
    if not isinstance(texto, str): 
        return ''
    
    #normalizar e remover acentos - NFKD 
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    #converter para minusculas
    texto_minusculo = texto_sem_acentos.lower()
    #texto limpo
    texto_limpo = re.sub(r'[^a-z\s]', '', texto_minusculo)  #sub faz substituição de padrões especificos no texto, ai substitui por vazio
    #remover espaços
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    
    return texto_limpo


data_base['texto_limpo'] = data_base['texto_review'].apply(limpa_texto)

print(data_base.sample(10))

#=========================================================
#Engenharia de Atributos - Converter sentimento em 0, 1
#=========================================================

data_base['sentimento_rotulo'] = data_base['sentimento'].map(({'positivo': 1, 'negativo': 0}))
print(data_base.sample(10))


#=====================================
#Dividir variáveis de treino e teste
#=====================================

#1. dividir variáveis de entrada(X - texto) e saída Y(sentimento)
x = data_base['texto_limpo']
y = data_base['sentimento_rotulo']

#2. dividir dados em treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size=0.25, random_state= 42, stratify= y)

#===================================
#Modelagem preditiva -  Pipeline
#===================================

#1. Texto em números através de vetor: tfidfVectorizer
#2. Padronização
#3. MachineLearning

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=['de', 'a', 'o', 'e', 'que', 'do', 'da', 'em', 'na', 'no', 'um','uns', 'uma'])),
    ('scaler', StandardScaler(with_mean=False)),
    ('logreg', LogisticRegression(solver= 'liblinear',random_state= 42, max_iter=1000))
    ])

print(type(pipeline))


#4. Otimizar os hiperparametros 

#Grid de hiperparametros
parametros_grid = {
    'tfidf__max_features':[500, 1000, 2000],
    'tfidf__ngram_range': [(1,1),(1,2)],
    'logreg__C':[0.1, 1, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__max_iter': [5000, 6000]
    }

#Configurar o grid com o pipeline

grid_search = GridSearchCV(
    pipeline, #pre processamento e modelo
    parametros_grid, #hiperparametros para teste 
    cv = 5, #divisões para validacao
    n_jobs = -1, #acelera processo
    scoring = 'accuracy', 
    verbose = 1 #detalhamento da execução
)

#Treinamento do modelo
print('-'*50)
print('TREINANDO O MODELO'.center(50))
grid_search.fit(X_treino, Y_treino)

print('-'*50)
print('MELHORES HIPERPARÂMETROS'.center(50))
print(grid_search.best_params_)

modelo = grid_search.best_estimator_ #recebe o modelo com melhores parametros

#========================
#5. Testes no modelo
#========================

#Previsão do conjunto de teste 
y_previsao = modelo.predict(X_teste)

#Calcular métricas de avaliação da previsao

acuracia = accuracy_score(Y_teste, y_previsao)
reporte = classification_report(Y_teste, y_previsao, target_names = ['Negativo', 'Positivo'])

print('-'*50)
print(f'Acurácia do Modelo: {acuracia:.2%}' )
print('Relatório de Classificação'.center(50))
print(reporte)

#===========================
#Matriz de confusão
#===========================

matriz = confusion_matrix(Y_teste, y_previsao)
seaborn.heatmap(matriz, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Negativo', 'Positivo'],
    yticklabels = ['Negativo', 'Positivo']
)
matplotlib.pyplot.title('Matriz de Confusão')
matplotlib.pyplot.xlabel('Previsão')
matplotlib.pyplot.ylabel('Verdadeiro')
matplotlib.pyplot.show()

#SALVAR MODELO

joblib.dump(modelo, 'modelo_sentimento.joblib')


