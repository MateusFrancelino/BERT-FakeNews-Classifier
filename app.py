from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import requests
import torch



app = Flask(__name__)

# URL do arquivo .pth no GitHub
model_url = "https://huggingface.co/josiscreydisom/FakeNewsDetectionBert8BG15e/resolve/main/modelo.pth"

# Faça o download do arquivo .pth
response = requests.get(model_url)

if response.status_code == 200:
    with open("modelo.pth", 'wb') as file:
        file.write(response.content)

# Define the path to the model and tokenizer you want to use
model_path = r"C:\Modelo\GradientAtivo\10e8batch\5e-5\modelo.pth"
tokenizer_path = r"C:\Modelo\GradientAtivo\10e8batch\5e-5\ModeloTreinado"


model = torch.load("modelo.pth")
model.to('cuda:0')

tokenizer = AutoTokenizer.from_pretrained("josiscreydisom/FakeNewsDetectionBert8BG15e")


# Função para buscar no Bing
def search_bing(query, api_key='6c08fc90ce6f404e914c0ed96b084f76'):
    # URL base da API de Pesquisa da Web do Bing
    base_url = 'https://api.bing.microsoft.com/v7.0/search'

    # Parâmetros da consulta
    params = {
        'q': query,
        'count': 4,  # Número de resultados desejados
        'mkt': 'pt-BR'
    }

    # Cabeçalhos da requisição
    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
    }

    # Fazer a requisição GET para a API
    response = requests.get(base_url, params=params, headers=headers)

    # Verificar o código de resposta
    if response.status_code == 200:
        # Obter os resultados da pesquisa
        search_results = response.json()

        # Processar e retornar os títulos e links dos resultados
        try:
            results = []
            for result in search_results['webPages']['value']:
                title = result['name']
                link = result['url']
                results.append({'Titulo': title, 'Link': link})
            return results
        except KeyError:
            return [{'Título': 'Não há resultados de pesquisa disponíveis.', 'Link': ''}]
    else:
        return [{'Título': 'Erro na requisição: ' + str(response.status_code), 'Link': ''}]

# Função para sumarizar o texto
def summarize_text(text, num_sentences=2):
    # Tokenização do texto em sentenças
    sentences = sent_tokenize(text)

    # Remoção de palavras irrelevantes (stop words)
    stop_words = set(stopwords.words("portuguese"))
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]

    # Cálculo da frequência das palavras
    word_frequency = FreqDist(words)

    # Atribuição de peso a cada sentença com base na frequência das palavras
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequency.keys():
                if len(sentence.split(" ")) < 30:  # Evitar penalizar sentenças muito longas
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequency[word]
                    else:
                        sentence_scores[sentence] += word_frequency[word]

    # Ordenar as sentenças com base nos pesos
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # Selecionar as melhores sentenças para o resumo
    summary_sentences = [sentence for sentence, score in sorted_sentences[:num_sentences]]

    # Juntar as sentenças selecionadas em um resumo
    summary = " ".join(summary_sentences)

    return summary

# Função para consultar fatos
def consultar_fatos(texto):
    url = f'https://factchecktools.googleapis.com/v1alpha1/claims:search?query={texto}&key=AIzaSyB9hocWXBfZ6OqtNZDjsu28UO4RP-Y0DXg'

    try:
        resposta = requests.get(url)
        resposta_json = resposta.json()

        if 'claims' in resposta_json:
            # Iterar sobre os resultados
            for resultado in resposta_json['claims']:
                return resultado
        else:
            return None

    except requests.exceptions.RequestException as e:
        print('Ocorreu um erro na solicitação:', str(e))

        
def create_response(fact_result, predicted_label=None):
    response_data = {
        'googlefactcheck': [],
        'bertimbauclassifier': [{
            'binaryrating': predicted_label,
            'texturalRating': 'Falso' if predicted_label else 'Verdadeiro',
            'result': 'O texto inserido foi classificado como falso' if predicted_label else 'O texto inserido foi classificado como verdadeiro'
        }]
    }

    if fact_result:
        response_data['googlefactcheck'].append({
            'verified_by': fact_result['claimReview'][0]['publisher']['site'],
            'url': fact_result['claimReview'][0]['url'],
            'reviewDate': fact_result['claimReview'][0]['reviewDate'],
            'textualRating': fact_result['claimReview'][0]['textualRating'],
            'title': fact_result['claimReview'][0]['title']
        })

    return jsonify(response_data)

@app.route('/')
def hello():
    return "O app está executando!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        input_text = data['text']
        

        # Tokenizar
        inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True, max_length=500, padding="max_length", truncation=True)
        inputs.to('cuda:0')
        outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        print(probabilities)
        #Determinando a classificação
        predicted_label = 1 if probabilities[0][1] > probabilities[0][0] else 0

        

        
        summary = summarize_text(input_text)
        print(search_bing(summary))

        fact_result = consultar_fatos(summary)

        if fact_result is None:
            search_results = search_bing(summary)
            for result in search_results:
                fact_result = consultar_fatos(result['Link'])
                if fact_result:
                    return create_response(fact_result)
        else:
            return create_response(fact_result, predicted_label)
        
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
