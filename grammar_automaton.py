import spacy

nlp = spacy.load("pt_core_news_sm")

conjuncoes_oracao = {'mas', 'porém', 'contudo', 'todavia'}
conjuncoes_termos = {'e', 'ou'}

def classificar_palavra(palavra):
    lower = palavra.lower()
    if lower in ('a', 'o'):
        return 'DET'

    doc = nlp(palavra)
    token = doc[0]

    # particípio passado tratado como ADJ
    if token.pos_ == 'VERB' and token.morph.get('VerbForm') == ['Part']:
        return 'ADJ'

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
            'ADP': 'complemento',   
            'CCONJ_TERMO': 'complemento',
            'CCONJ_ORACAO': 'inicio',
            'VERB': 'verbo',
            'AUX': 'verbo'
        }
    }
    estados_aceitacao = ['verbo', 'adv', 'complemento']
    return automato, estados_aceitacao
    
def verificar_gramatica(frase: str) -> bool:
    doc = nlp(frase)
    # filtra pontuação, mantém tokens
    palavras = [token.text for token in doc if not token.is_punct]

    automato, estados_aceitacao = criar_automato()
    estado_atual = 'inicio'

    for palavra in palavras:
        pos = classificar_palavra(palavra)

        # trata conjunções
        if pos == 'CCONJ':
            lower = palavra.lower()
            if lower in conjuncoes_oracao:
                pos = 'CCONJ_ORACAO'
            elif lower in conjuncoes_termos:
                pos = 'CCONJ_TERMO'
            else:
                print(f'Conjunção "{palavra}" não categorizada. Frase errada.')
                return False

        print(f'Palavra: {palavra}, Classe: {pos}, Estado atual: {estado_atual}')

        trans = automato.get(estado_atual, {})
        if pos in trans:
            estado_atual = trans[pos]
        else:
            print(f'Frase errada! A palavra "{palavra}" não se encaixa em uma transição válida no estado "{estado_atual}".')
            return False

    if estado_atual in estados_aceitacao:
        return True
    else:
        print(f'Frase errada! Finalizou em estado: {estado_atual}')
        return False