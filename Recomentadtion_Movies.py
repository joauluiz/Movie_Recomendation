import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
import sys
import re
import time
import json

warnings.filterwarnings("ignore", category=Warning)

#Importando os dados do filme e suas avaliações. O parâmetro low_memory é usado pois ele identifica os tipos das variáveis da coluna, que para esse caso estamos deixando False
Data_filmes =  pd.read_csv("data/movies_metadata.csv", low_memory=False)
Data_avali = pd.read_csv("data/ratings.csv",low_memory=False)

Data_filmes = Data_filmes[['adult','budget', 'genres', 'original_language', 'id', 'revenue','spoken_languages','vote_average', 'vote_count', 'original_title']]
Data_avali = Data_avali[['userId', 'movieId', 'rating']]

#Renomeando Colunas (Não era um processo necessário, mas estou fazendo mais para praticar)
Data_filmes.rename(columns={'id':'ID FILME','genres':'Gênero','Budget':'Orçamento','vote_count':'Quantidade Avaliações'},inplace=True)
Data_avali.rename(columns={'userId':'ID Usuário', 'movieId':'ID FILME', 'rating':'Avaliação'},inplace=True)

#Tirando linhas com colunas vazias
Data_filmes.dropna(inplace=True)
Data_avali.dropna(inplace=True)



#REGRA DE NEGÓCIO. Iremos considerar apenas avaliações em que o usuário tem mais de 100 avaliações.
#Essas regras podem mudar dependendo do negócio
#Isso irá gerar uma série que contarará quantas vezes cada usuário já fez uma avaliação. Retorna uma série booleana.
Qntds_Avali = Data_avali['ID Usuário'].value_counts() > 1000


#Será considerado apenas os valores TRUE, retornando os valores do índice
y = Qntds_Avali[Qntds_Avali].index

#Agora que já selecionamos todos os usuários que devemos considerar suas avaliações, é necessário que o df original contenha apenas
#esses usuários, para isso utilizaremos o método isin()
Data_avali = Data_avali[Data_avali['ID Usuário'].isin(y)]

#Irei realizar os mesmo passos agora com o dataset de filmes, selecionando os filmes que tem mais de 200 avaliações
Data_filmes = Data_filmes[Data_filmes['Quantidade Avaliações']>200]

#Verificando os tipos das variáveis em ambas as tabelas, antes de juntar elas
# print(Data_filmes.info())
# print(Data_avali.info())

#Mudando o formato da coluna ID FILME do df de filmes para deixar no mesmo formato int que a coluna id filme do df de avaliação
#para que seja possível fazer um merge
Data_filmes['ID FILME']=Data_filmes ['ID FILME'].astype(int)

#Juntando ambos os dataframes de filmes de avaliações em um só, sendo necessário que ambas as colunas tenham o mesmo nome
Avali_Filmes = Data_avali.merge(Data_filmes, on = 'ID FILME')


#Em um caso de um mesmo usuário ter avaliado o mesmo filme mais de uma vez, faz-se necessário com que seja retirada as informações duplicadas
Avali_Filmes.drop_duplicates(['ID Usuário','ID FILME'], inplace=True)
Filmes = Data_filmes['original_title']

#Podemos retirar a coluna do ID do filme, que foi utilizada anteriormente para fazer o merge entre as colunas
del Avali_Filmes['ID FILME']

#Agora é necessário transformar os usuários como uma variável (transformar em coluna), para isso precisamos PIVOTEAR as linhas para coluna.
#Iremos fazer com que a coluna índice seja o título do filme, as colunas os ids dos usuários e os valores mostrados as notas dadas
#desta forma poderemos ver uma visão melhor das notas
filmes_pivotados = Avali_Filmes.pivot_table (columns = 'ID Usuário', index = 'original_title', values = 'Avaliação')
#print(filmes_pivotados.loc['Turist'])

#Trocando os valores NAN por 0, para que seja possível entrar na rede neural
filmes_pivotados.fillna(0, inplace=True)

#Agora será necessário criar uma matriz esparsa  devido à grande quantidade de 0, o que seria muito custoso computacionalmente
#Isso fará com que a matriz fique compactada
filmes_pivotados_sparse = csr_matrix(filmes_pivotados)

#Criação do modelo KNN
modelo = NearestNeighbors(algorithm = 'brute')
modelo.fit(filmes_pivotados_sparse)

#Criando um array para ir anexando as escolhar do filme

Recomendacoes_Filmes =[]
val=0

while True:
    print('Escolha um dos filmes que você já assistiu para obter recomendações de filmes relacionados (digite 0 para sair):',Filmes)
    Escolha_Filme = input ("Digite:")
    if Escolha_Filme != "0":
        try:
            distancia, sugestoes = modelo.kneighbors(filmes_pivotados.filter(items=[Escolha_Filme], axis=0).values.reshape(1, -1))
            if val == 0:
                Recomendacoes_Filmes.append(np.delete(filmes_pivotados.index[sugestoes],0))
                val = 1
            elif val != 0:
                val_sugestoes = np.delete(filmes_pivotados.index[sugestoes],0)
                Recomendacoes_Filmes.append(val_sugestoes)
            print(filmes_pivotados.index[sugestoes])
        except ValueError:
            print(ValueError)
            print("Não foi possível encontrar filmes relacionados para a escolha selecionada.")
            print(sys.exc_info()[1])
    else:
        break

Genero=[]

#Transformando as respostas do array em elementos de uma lista
Lista_Recomendação = np.array(Recomendacoes_Filmes).flatten().tolist()

Genero = Data_filmes[Data_filmes['original_title'].isin(Lista_Recomendação)]


tam =[]

for i in range (len(Genero['Gênero'])):
    ##COLOCAR O CORREDOR DE STRING
    teste = Genero['Gênero'].iloc[i]
    matches = re.finditer(r"name': '(.*?)'", teste)
    for match in matches:
        tam.append(match.group(1))


var = np.empty((len(tam),2), dtype=object)


#Depois transformar os generos em uma lista



# DEPOIS EU PRECISO FAZER UM FOR DENTRO DO OUTRO, SENDO O DE FORA PARA ALTERAR O FILME, E O DE DENTRO PARA COLOCAR NA MATRIZ
sum = -1
for i in range (len(Genero['Gênero'])):
    filme = Genero['original_title'].iloc[i]
    matches = re.findall(r"name': '(.*?)'",Genero['Gênero'].iloc[i])
    for j in range (len(matches)):
        sum = sum + 1
        var[sum,0] = filme
        var[sum,1] = tam[sum]

var = pd.DataFrame(var,columns=["Filmes","Gêneros"])

while True:
    print("Digite o gênero no qual deseja ter as recomendações (Digite 0 para terminar o código) ")
    unique = var["Gêneros"].unique()
    print("Essas são as listas de gêneros disponíveis: ", unique)
    Digit = input("Digite:")
    if Digit != "0":
        print("Essas são as recomendações para o gênero ",Digit,":")
        mask = var.loc[var['Gêneros']== Digit]
        print(mask['Filmes'].to_string())
        time.sleep(10)
    else:
        break

print("Fim do código. Muito obrigado")



