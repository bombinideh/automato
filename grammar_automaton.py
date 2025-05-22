import spacy

nlp = spacy.load("pt_core_news_sm")

conjuncoes_oração = {'mas', 'porém', 'contudo', 'todavia'}
conjuncoes_termos = {'e', 'ou'}

def classificar_palavra(palavra):
    if palavra.lower() == 'a' or palavra.lower() == 'o':
        return 'DET'  # Corrige caso específico
    doc = nlp(palavra)
    for token in doc:
        return token.pos_

def criar_automato():
    automato = {
        'inicio': {
            'DET': 'sujeito',
            'PRON': 'sujeito',
            'NOUN': 'sujeito',
        },
        'sujeito': {
            'DET': 'sujeito',
            'NOUN': 'sujeito',
            'PRON': 'sujeito',
            'ADJ': 'sujeito',
            'ADV': 'sujeito',
            'VERB': 'verbo',
            'AUX': 'verbo'
        },
        'verbo': {
            'VERB': 'verbo',
            'AUX': 'verbo',
            'ADV': 'adv',
            'ADP': 'adp',
            'DET': 'complemento',
            'NOUN': 'complemento',
            'PRON': 'complemento',
            'ADJ': 'complemento',
            'CCONJ_TERMO': 'complemento',
            'CCONJ_ORACAO': 'inicio'
        },
        'adv': {
            'ADV': 'adv',
            'ADP': 'adp',
            'DET': 'complemento',
            'NOUN': 'complemento',
            'PRON': 'complemento',
            'ADJ': 'complemento',
            'CCONJ_TERMO': 'complemento',
            'CCONJ_ORACAO': 'inicio'
        },
        'adp': {
            'DET': 'complemento',
            'NOUN': 'complemento',
            'PRON': 'complemento',
            'ADJ': 'complemento',
            'CCONJ_TERMO': 'complemento',
            'CCONJ_ORACAO': 'inicio'
        },
        'complemento': {
            'NOUN': 'complemento',
            'PRON': 'complemento',
            'ADJ': 'complemento',
            'DET': 'complemento',
            'ADV': 'complemento',
            'ADP': 'complemento',   # permite preposições aqui
            'CCONJ_TERMO': 'complemento',
            'CCONJ_ORACAO': 'inicio'
        }
    }
    estados_aceitacao = ['verbo', 'adv', 'complemento']
    return automato, estados_aceitacao

def verificar_gramatica(frase):
    doc = nlp(frase)
    palavras = [token.text for token in doc if not token.is_punct]

    automato, estados_aceitacao = criar_automato()
    estado_atual = 'inicio'

    for palavra in palavras:
        posicao = classificar_palavra(palavra)
        
        if posicao == 'CCONJ':
            if palavra.lower() in conjuncoes_oração:
                chave = 'CCONJ_ORACAO'
            elif palavra.lower() in conjuncoes_termos:
                chave = 'CCONJ_TERMO'
            else:
                print(f'Conjunção "{palavra}" não categorizada. Frase errada.')
                return False
            posicao = chave
        
        print(f'Palavra: {palavra}, Classe: {posicao}, Estado atual: {estado_atual}')
        
        if posicao in automato.get(estado_atual, {}):
            estado_atual = automato[estado_atual][posicao]
        else:
            print(f'Frase errada! A palavra "{palavra}" não se encaixa em uma transição válida no estado "{estado_atual}".')
            return False

    if estado_atual in estados_aceitacao:
        # print('Frase correta!')
        return True
    else:
        print(f'Frase errada! Finalizou em estado: {estado_atual}')
        return False