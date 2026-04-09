# %%


# %%
#imports
import sys
import os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
import PIL
import random
random.seed(100)
np.random.seed(100)

# Visualização de Dados e Imagens Médicas
import matplotlib.image as mpimg

import tensorflow as tf

import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model


from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Dropout

import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# #**Gestão de Diretórios e Caminhos de Dados**

# %%
#Obtém o caminho absoluto do diretório onde o notebook está sendo executado
current_working_directory = os.getcwd()

#Exibe o caminho no console para verificação
print(current_working_directory)

# %%
#Aquisição e carregamento para metadados
#Definição do caminho absoluto para o arquivo de metadados (CSV)
csv_path = 'C:/Users/analice/Documents/TCC/dataset/csv/meta.csv'

#Carregamento do arquivo CSV utilizando a biblioteca Pandas
df_meta = pd.read_csv(csv_path)

#Visualização das primeiras linhas
df_meta

# %%
#O arquivo 'dicom_info.csv' funciona como a ponte entre os dados clínicos e a localização física dos arquivos no armazenamento
dicom_data = pd.read_csv('C:/Users/analice/Documents/TCC/dataset/csv/dicom_info.csv')

#Exibição do cabeçalho para validar colunas críticas
dicom_data.head()

# %%
#Análise exploratória de dados
#O método .info() fornece um resumo estrutural do dataframe
dicom_data.info()

# %%
#Identificação dos tipos de imagens
#O método .unique() lista todos os valores distintos na coluna 'SeriesDescription'
dicom_data.SeriesDescription.unique()

# %%
#Mapeamento de tratamento de caminhos de arquivos
#Define a raiz de onde as imagens JPEG estão armazenadas no meu Google Drive
image_dir = 'C:/Users/analice/Documents/TCC/dataset/jpeg'

#Separa os caminhos das imagens em três categorias distintas baseadas na descrição técnica:
#1. Imagens completas da mama.
#2. Imagens recortadas (focadas na lesão).
#3. Máscaras de Segmentação (ROI)
full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

# Ajusta os caminhos originais do CSV para apontar para a pasta no Drive.
full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
roi_mask_images = roi_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

#Exibe o primeiro caminho da lista para verificar se a substituição ocorreu corretamente
full_mammogram_images.iloc[0]

# %%
#O atributo .shape retorna o número de caminhos de arquivos
full_mammogram_images.shape

# %%
#O método .iloc[0] acessa o primeiro elemento da série 'cropped_images'
cropped_images.iloc[0]

# %%
#O atributo .shape retorna o número total de amostras de imagens recortadas (cropped)
cropped_images.shape

# %%
#O método .iloc[0] acessa o primeiro caminho da lista 'roi_mask_images'
roi_mask_images.iloc[0]

# %%
#O atributo .shape retorna o total de arquivos de máscaras (ROI) encontrados
roi_mask_images.shape

# %%
# Inicializa dicionários para busca rápida de caminhos
full_mammogram_dict = dict()
cropped_dict = dict()
roi_mask_dict = dict()

#Itera sobre os caminhos das imagens completas
for dicom in full_mammogram_images:
    # print(dicom)
    key = dicom.split("/")[7]
    # print(key)
    full_mammogram_dict[key] = dicom

#Itera sobre os recortes (crops) e mapeia para o dicionário correspondente
for dicom in cropped_images:
    key = dicom.split("/")[7]
    cropped_dict[key] = dicom

# Itera sobre as máscaras (ROI) e mapeia para o dicionário correspondente
for dicom in roi_mask_images:
    key = dicom.split("/")[7]
    roi_mask_dict[key] = dicom

# %%
#Validação do primeiro mapeamento de dicionario
next(iter((full_mammogram_dict.items())))

# %%
#Monitoramento de memoria RAM
#O método sys.getsizeof() retorna o tamanho do objeto (dicionário) em byteS
sys.getsizeof(full_mammogram_dict)

# %%
#Extrai o primeiro par (chave: valor) do dicionário de imagens recortadas
next(iter((cropped_dict.items())))

# %%
#Analise de consumo de memoria
#O método sys.getsizeof() retorna o tamanho do objeto 'cropped_images' em bytes
sys.getsizeof(cropped_images)

# %%
#Validação do mapeamento de mascaras
#Extrai o primeiro par (chave: valor) do dicionário de máscaras de segmentação
next(iter((roi_mask_dict.items())))

# %%
#Analise de consumo de memoria
#O método sys.getsizeof() retorna o tamanho em bytes da Série 'roi_mask_images'
sys.getsizeof(roi_mask_images)

# %%
#Carregamento das descrições clinicas e rotulos

#Carrega os dados de treinamento e teste para casos de MASSAS
mass_train_data = pd.read_csv('C:/Users/analice/Documents/TCC/dataset/csv/mass_case_description_train_set.csv')
mass_test_data = pd.read_csv('C:/Users/analice/Documents/TCC/dataset/csv/mass_case_description_test_set.csv')

#Carrega os dados de treinamento e teste para casos de CALCIFICAÇÕES
calc_train_data = pd.read_csv('C:/Users/analice/Documents/TCC/dataset/csv/calc_case_description_train_set.csv')
calc_test_data = pd.read_csv('C:/Users/analice/Documents/TCC/dataset/csv/calc_case_description_test_set.csv')

# %%
#O método .head() exibe as 5 primeiras linhas do conjunto
mass_train_data.head()

# %%
mass_test_data.head()

# %%
calc_train_data.head()

# %%
calc_test_data.head()

# %%
#Distribuição de patologias

#Conta a frequência de cada categoria no conjunto de TREINO de massas
train_counts = mass_train_data['pathology'].value_counts()

#Conta a frequência de cada categoria no conjunto de TESTE de massas
test_counts = mass_test_data['pathology'].value_counts()

#Exibição dos resultados
print("Distribuição no Conjunto de Treino (Massas):")
print(train_counts)

print("\nDistribuição no Conjunto de Teste (Massas):")
print(test_counts)

# %%
#Realiza a contagem de frequência das categorias patológicas para os casos de CALCIFICAÇÕES no treino
train_counts = calc_train_data['pathology'].value_counts()

# Realiza a contagem para o conjunto de teste de calcificações
test_counts = calc_test_data['pathology'].value_counts()

# Exibição dos resultados
print("Distribuição no Conjunto de Treino (Calcificações):")
print(train_counts)

print("\nDistribuição no Conjunto de Teste (Calcificações):")
print(test_counts)

# %%
#Função de correção e vinculação de caminhos (MASSAS)

def fix_image_path_mass(dataset):
    """
    Esta função percorre o DataFrame e substitui os caminhos de arquivos originais (inválidos)
    pelos caminhos absolutos corretos armazenados no Google Drive
    """
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        if img_name in full_mammogram_dict:
            dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        img_name = img[12].split("/")[2]
        if img_name in cropped_dict:
            dataset.iloc[i, 12] = cropped_dict[img_name]

        img_name = img[13].split("/")[2]
        if img_name in roi_mask_dict:
            dataset.iloc[i, 13] = roi_mask_dict[img_name]

# %%
img_name = mass_train_data.iloc[0, 11].split("/")[1]

print("Extraído:", img_name)
print("Existe no dict?", img_name in full_mammogram_dict)

# %%
print("Extraído:", list(full_mammogram_dict.values())[0])

print("Extraído:",list(full_mammogram_dict.keys())[0])

# %%
#Chama a função fix_image_path_mass passando o DataFrame de treinamento
#Este comando percorre todas as linhas do 'mass_train_data' e:
#1. Localiza o ID da imagem na coluna de caminhos original
#2. Busca o endereço físico correspondente no seu Google Drive (via dicionários)
#3. Sobrescreve o DataFrame com o caminho absoluto final
fix_image_path_mass(mass_train_data)

# %%
fix_image_path_mass(mass_test_data)

# %%
mass_train_data

# %%
mass_test_data

# %%
#Mesma coisa que a função fix_image_path_mass só que com as calcificações
def fix_image_path_calc(dataset):
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        if img_name in full_mammogram_dict:
            dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        img_name = img[12].split("/")[2]
        if img_name in cropped_dict:
            dataset.iloc[i, 12] = cropped_dict[img_name]

        img_name = img[13].split("/")[2]
        if img_name in roi_mask_dict:
            dataset.iloc[i, 13] = roi_mask_dict[img_name]

# %%
fix_image_path_calc(calc_train_data)

# %%
fix_image_path_calc(calc_test_data)

# %%
calc_train_data

# %%
calc_test_data

# %%
#O método .unique() extrai todos os valores distintos presentes na coluna 'pathology'
mass_train_data.pathology.unique()

# %%
calc_train_data.pathology.unique()

# %%
mass_train_data.info()

# %%
calc_train_data.info()

# %%
#Normalização e Renomeio das colunas

#O método .rename() substitui nomes com espaços por nomes com 'underscores' (_)
mass_train = mass_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_train.head()

# %%
calc_train = calc_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                             'breast density':'breast_density',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'calc type': 'calc_type',
                                           'calc distribution': 'calc_distribution',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

calc_train.head()

# %%
#O método .isnull().sum() percorre todas as colunas do DataFrame 'mass_train' e:
#1. Identifica células vazias ou nulas (NaN)
#2. Soma a quantidade de nulos por coluna
mass_train.isnull().sum()

# %%
calc_train.isnull().sum()

# %%
#Tratamento de dados Faltantes
#O método 'bfill' (backward fill) preenche os valores nulos (NaN) com o próximo valor
mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

#Verifica novamente a soma de valores nulos
mass_train.isnull().sum()

# %%
calc_train['calc_type'] = calc_train['calc_type'].fillna(method='bfill')
calc_train['calc_distribution'] = calc_train['calc_distribution'].fillna(method='bfill')

calc_train.isnull().sum()

# %%
mass_test_data.isnull().sum()

# %%
calc_test_data.isnull().sum()

# %%
print(mass_test_data.columns,'\n')

mass_test = mass_test_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_test.columns

# %%
print(calc_test_data.columns,'\n')

calc_test = calc_test_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'breast density':'breast_density',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'calc type': 'calc_type',
                                           'calc distribution': 'calc_distribution',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

calc_test.columns

# %%
calc_test['calc_type'] = calc_test['calc_type'].fillna(method='bfill')
calc_test['calc_distribution'] = calc_test['calc_distribution'].fillna(method='bfill')

calc_test.isnull().sum()

# %% [markdown]
# # **SUMARIZAÇÃO QUANTITATIVA DAS CARACTERÍSTICAS (FEATURES)**

# %%
#O método .describe() calcula estatísticas fundamentais para colunas numéricas:
#- count: Total de registros (ajuda a ver se há nulos remanescentes)
#- mean: Média aritmética (ex: média da densidade mamária)
#- std: Desvio padrão (indica a variabilidade dos dados)
#- min/max: Valores extremos
#- 25%, 50%, 75%: Percentis (mediana e quartis)
mass_train.describe()

# %%
calc_train.describe()

# %%
#O atributo .shape retorna uma tupla (linhas, colunas)
print(f'Shape of mass_train: {mass_train.shape}')
print(f'Shape of mass_test: {mass_test.shape}')

# %%
print(f'Shape of calc_train: {calc_train.shape}')
print(f'Shape of calc_test: {calc_test.shape}')

# %% [markdown]
# #**VISUALIZAÇÃO DA DISTRIBUIÇÃO GLOBAL DE PATOLOGIAS (TREINO)**

# %%
#Soma as contagens de patologia de ambos os conjuntos de treino (Massas e Calcificações)
value = mass_train['pathology'].value_counts() + calc_train['pathology'].value_counts()

#Geração do Gráfico de Pizza
plt.pie(value, labels=value.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribuição Total de Patologias (Massas + Calcificações)', fontsize=12)
plt.savefig('distribuicao_patologias_total.png', bbox_inches='tight')

# %%
# Define a paleta de cores para o conjunto de massas
mass_palette = sns.color_palette("crest", n_colors=len(mass_train['assessment'].unique()))

# Gera o gráfico de contagem cruzando Avaliação (BI-RADS) com Patologia
sns.countplot(data=mass_train, y='assessment', hue='pathology', palette=mass_palette)

# Tradução das legendas e títulos
plt.title('Distribuição de Avaliações - Treinamento de Massas')
plt.xlabel('Contagem')
plt.ylabel('Avaliação (Assessment)')
plt.legend(title='Patologia') # Traduz a legenda lateral
plt.show()

# Define a paleta de cores para o conjunto de calcificações
calc_palette = sns.color_palette("magma", n_colors=len(calc_train['assessment'].unique()))

# Gera o gráfico de contagem
sns.countplot(data=calc_train, y='assessment', hue='pathology', palette=calc_palette)

# Tradução das legendas e títulos
plt.title('Distribuição de Avaliações - Treinamento de Calcificações')
plt.xlabel('Contagem')
plt.ylabel('Avaliação (Assessment)')
plt.legend(title='Patologia')
plt.show()

# %%
#ANÁLISE DE SUTILEZA DAS MASSAS (SUBTLETY)

plt.figure(figsize=(8, 6))

# Gera o gráfico de contagem para o Grau de Sutileza
# No CBIS-DDSM, 'subtlety' indica o quão difícil é ver a lesão (1 é muito sutil, 5 é óbvia).
sns.countplot(data=mass_train, x='subtlety', palette='magma', hue='subtlety', legend=False)

# Tradução dos elementos do gráfico para o Português
plt.title('Sutileza das Massas de Câncer de Mama', fontsize=14, fontweight='bold')
plt.xlabel('Grau de Sutileza (Subtlety)', fontsize=12)
plt.ylabel('Contagem de Casos', fontsize=12)

# Exibe o gráfico
plt.show()

# %%
# --- ANÁLISE DE SUTILEZA DAS CALCIFICAÇÕES (SUBTLETY) ---

plt.figure(figsize=(8, 6))

# Gera o gráfico de contagem para o Grau de Sutileza das Calcificações
# Usando a paleta 'viridis' conforme solicitado
sns.countplot(data=calc_train, x='subtlety', palette='viridis', hue='subtlety', legend=False)

# Tradução dos elementos do gráfico para o Português
plt.title('Sutileza das Calcificações de Câncer de Mama', fontsize=14, fontweight='bold')
plt.xlabel('Grau de Sutileza (Subtlety)', fontsize=12)
plt.ylabel('Contagem de Casos', fontsize=12)

# Exibe o gráfico final
plt.show()

# %%
# --- ANÁLISE DE FORMA DA MASSA VS. PATOLOGIA ---

plt.figure(figsize=(10, 6))

# Gera o gráfico de contagem comparando o formato da massa com o diagnóstico (Patologia)
# O parâmetro 'hue' permite ver a proporção de Malignos/Benignos para cada formato.
sns.countplot(data=mass_train, x='mass_shape', hue='pathology', palette='viridis')

# Tradução dos elementos do gráfico para o Português
plt.title('Distribuição do Formato da Massa por Patologia', fontsize=14, fontweight='bold')
plt.xlabel('Formato da Massa (Mass Shape)', fontsize=12)
plt.ylabel('Contagem de Casos', fontsize=12)

# Ajusta a rotação das legendas do eixo X para não sobrepor os nomes (ex: Irregular, Oval)
plt.xticks(rotation=30, ha='right')

# Traduz o título da legenda lateral
plt.legend(title='Patologia')

# Exibe o gráfico final
plt.tight_layout() # Garante que os rótulos não sejam cortados ao salvar
plt.show()

# %%
# --- ANÁLISE DE TIPO DE CALCIFICAÇÃO VS. PATOLOGIA ---

plt.figure(figsize=(12, 8))

# Gera o gráfico de contagem comparando o tipo morfológico da calcificação com a Patologia
# O uso de y='calc_type' cria um gráfico horizontal, ideal para nomes longos de tipos.
sns.countplot(data=calc_train, y='calc_type', hue='pathology', palette='viridis')

# Tradução dos elementos do gráfico para o Português
plt.title('Distribuição do Tipo de Calcificação por Patologia', fontsize=14, fontweight='bold')
plt.xlabel('Contagem de Casos (Patologia)', fontsize=12)
plt.ylabel('Tipo de Calcificação (Calc Type)', fontsize=12)

# Ajuste dos rótulos do eixo Y (alinhamento à direita para leitura clara)
plt.yticks(rotation=0, ha='right')

# Move a legenda para fora do gráfico para não sobrepor as barras
plt.legend(title='Patologia', loc='upper right', bbox_to_anchor=(1.25, 1))

# Exibe o gráfico final
plt.tight_layout()
plt.show()

# %%
# --- ANÁLISE DE DENSIDADE MAMÁRIA VS. PATOLOGIA ---

plt.figure(figsize=(10, 8))

# Gera o gráfico de contagem comparando a densidade da mama com o diagnóstico real
# 1: Gordurosa | 2: Densidades Fibroglandulares Esparsas
# 3: Heterogeneamente Densa | 4: Extremamente Densa
sns.countplot(data=mass_train, x='breast_density', hue='pathology', palette='viridis')

# Tradução dos títulos e eixos para o Português
plt.title('Densidade Mamária vs. Patologia\n\n1: Gordurosa || 2: Tecido Fibroglandular Esparso\n3: Heterogeneamente Densa || 4: Extremamente Densa',
          fontsize=12, fontweight='bold')

plt.xlabel('Graus de Densidade (BI-RADS)', fontsize=12)
plt.ylabel('Contagem de Casos', fontsize=12)

# Traduz o título da legenda lateral
plt.legend(title='Patologia')

# Ajuste fino para evitar que o título superior seja cortado
plt.tight_layout()
plt.show()

# %%
# --- ANÁLISE DE DENSIDADE MAMÁRIA VS. PATOLOGIA (CALCIFICAÇÕES) ---

plt.figure(figsize=(10, 8))

# Gera o gráfico de contagem comparando a densidade da mama com o diagnóstico (Patologia)
# 1: Gordurosa | 2: Tecido Fibroglandular Esparso
# 3: Heterogeneamente Densa | 4: Extremamente Densa
sns.countplot(data=calc_train, x='breast_density', hue='pathology', palette='viridis')

# Tradução dos títulos e eixos para o Português (Padrão TCC)
plt.title('Densidade Mamária vs. Patologia (Calcificações)\n\n1: Gordurosa || 2: Tecido Fibroglandular Esparso\n3: Heterogeneamente Densa || 4: Extremamente Densa',
          fontsize=12, fontweight='bold')

plt.xlabel('Graus de Densidade (BI-RADS)', fontsize=12)
plt.ylabel('Contagem de Casos', fontsize=12)

# Tradução da legenda lateral
plt.legend(title='Patologia')

# Ajuste para garantir que o título e legendas caibam na imagem salva
plt.tight_layout()
plt.show()

# %%
mass_train.head()

# %%
calc_train.head()

# %%
import matplotlib.image as mpimg

def display_images(column, number):
    """Exibe as imagens do dataset para validação visual"""

    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # Loop para percorrer as primeiras linhas e exibir as imagens
    for index, (i, row) in enumerate(mass_train.head(number_to_visualize).iterrows()):
        image_path = row[column]

        # Verifica se o arquivo físico existe no caminho mapeado
        if os.path.exists(image_path):
            image = mpimg.imread(image_path)
            ax = axes[index]
            ax.imshow(image, cmap='gray') # Exibe em escala de cinza (padrão mamográfico)

            # Título com a patologia traduzida (opcional se você já traduziu os dados)
            ax.set_title(f"Patologia: {row['pathology']}")
            ax.axis('off') # Remove os eixos para uma visualização mais limpa
        else:
            print(f"Arquivo não encontrado: {image_path}")

    plt.tight_layout()
    plt.show()

# --- EXECUÇÃO DA VISUALIZAÇÃO ---

print('--- DATASET DE TREINAMENTO (MASSAS) ---\n')

print('1. Mamografias Completas (Full Mammograms):')
display_images('image_file_path', 5)

print('\n2. Recortes das Lesões (Cropped Images):')
# Esta é a visão que o EfficientNet geralmente usa para focar no tumor
display_images('cropped_image_file_path', 5)

print('\n3. Máscaras da Região de Interesse (ROI Masks):')
# Mostra exatamente onde o radiologista demarcou a lesão
display_images('ROI_mask_file_path', 5)

# %%
import matplotlib.image as mpimg


def display_images(column, number):
    """displays images in the dataset"""
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # Loop through rows and display images
    for index, row in mass_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        print(image_path)
        # Check if the file exists
        if os.path.exists(image_path):
            image = mpimg.imread(image_path)
            ax = axes[index]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"{row['pathology']}")
            ax.axis('off')
        else:
            print(f"File not found: {image_path}")

    plt.tight_layout()
    plt.show()

print('Mass Training Dataset\n\n')
print('Full Mammograms:\n')
display_images('image_file_path', 5)
print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 5)
print('ROI Images:\n')
display_images('ROI_mask_file_path', 5)

# %%
# --- VISUALIZAÇÃO DO DATASET DE TREINAMENTO (CALCIFICAÇÕES) ---

def exibir_imagens_calc(coluna, quantidade):
    """Exibe uma sequência de imagens para validação visual das calcificações"""

    quantidade_visualizar = quantidade
    rows = 1
    cols = quantidade_visualizar
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # Itera sobre as primeiras linhas do conjunto de calcificações
    # Usamos enumerate para garantir que o índice do eixo (ax) seja sempre de 0 a quantidade-1
    for i, (index, row) in enumerate(calc_train.head(quantidade_visualizar).iterrows()):
        image_path = row[coluna]

        # Verifica se o mapeamento para o Drive funcionou (Arquivo existe?)
        if os.path.exists(image_path):
            image = mpimg.imread(image_path)
            ax = axes[i]
            ax.imshow(image, cmap='gray')

            # Título com o diagnóstico em Português
            ax.set_title(f"Patologia: {row['pathology']}")
            ax.axis('off')
        else:
            print(f"Arquivo não encontrado: {image_path}")

    plt.tight_layout()
    plt.show()

# --- EXECUÇÃO DA VISUALIZAÇÃO ---

print('--- DATASET DE TREINAMENTO (CALCIFICAÇÕES) ---\n')

print('1. Mamografias Completas (Full Mammograms):')
exibir_imagens_calc('image_file_path', 5)

print('\n2. Recortes das Calcificações (Cropped Images):')
# Essencial para ver a morfologia (pleomórfica, punctata, etc)
exibir_imagens_calc('cropped_image_file_path', 5)

print('\n3. Máscaras de Segmentação (ROI Masks):')
# Mostra a exata localização das microcalcificações
exibir_imagens_calc('ROI_mask_file_path', 5)


