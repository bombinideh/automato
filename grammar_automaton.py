import spacy

# Carregar o modelo de linguagem em português
nlp = spacy.load("pt_core_news_sm")

# Função para classificar as palavras
def classificar_palavra(palavra):
    doc = nlp(palavra)
    for token in doc:
        return token.pos_

# Função para criar o autômato de transições
def criar_automato():
    # Definindo os estados e as transições
    automato = {
        'inicio': {'NOUN': 'sujeito', 'PRON': 'sujeito', 'DET': 'sujeito'},  # Agora aceita determinantes (como "um")
        'sujeito': {'VERB': 'verbo'},  # Após sujeito, espera verbo
        'verbo': {'NOUN': 'complemento', 'PRON': 'complemento', 'ADJ': 'complemento', 'ADV': 'complemento', 'DET': 'complemento', 'ADP': 'complemento'},  # Aceita complementos como adjetivos, advérbios, determinantes e preposições
        'complemento': {'NOUN': 'complemento', 'PRON': 'complemento', 'ADJ': 'complemento', 'ADV': 'complemento', 'DET': 'complemento', 'ADP': 'complemento'}  # Complementos podem continuar
    }
    # Estados de aceitação (último estado válido)
    estados_aceitacao = ['complemento']
    
    return automato, estados_aceitacao

# Função para verificar a sequência da frase com base no autômato
def verificar_gramatica(frase):
    # Tokenizar a frase
    palavras = frase.split()
    
    # Criar o autômato e os estados de aceitação
    automato, estados_aceitacao = criar_automato()
    
    # Definindo o estado inicial do autômato
    estado_atual = 'inicio'
    
    # Percorrer as palavras e realizar as transições no autômato
    for palavra in palavras:
        posicao = classificar_palavra(palavra)  # Classificar a palavra com o spacy
        print(f'Palavra: {palavra}, Classe: {posicao}, Estado atual: {estado_atual}')
        # Verificar se a transição é válida
        if posicao in automato[estado_atual]:
            estado_atual = automato[estado_atual][posicao]  # Realizar a transição
        else:
            print(f'Frase errada! A palavra "{palavra}" não se encaixa em uma transição válida.')
            return False

    # Verificar se o autômato terminou em um estado de aceitação
    if estado_atual in estados_aceitacao:
        return True
    else:
        print(f'Frase errada! A frase não terminou corretamente. Finalizou em estado: {estado_atual}')
        return False

# Testes de exemplo
# frase1 = "Eu comprei um livro grande"
# frase2 = "Ela bonita canta bem"
# frase3 = "Eles correram rápido pela rua"
# frase4 = "Ela ama ler livros"
