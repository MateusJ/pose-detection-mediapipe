# Detecção de Pose com MediaPipe

Este projeto utiliza MediaPipe e OpenCV para detectar e analisar poses humanas em tempo real através da webcam, com foco na análise de movimentos como agachamentos.

## Funcionalidades

- Detecção de pose em tempo real usando a webcam
- Cálculo de ângulos articulares (joelho e quadril)
- Visualização das landmarks da pose
- Interface visual com feedback em tempo real

## Requisitos

- Python 3.7+
- Webcam funcional

## Instalação

1. Clone este repositório:
```bash
git clone https://github.com/SEU_USUARIO/pose-detection-mediapipe.git
cd pose-detection-mediapipe
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como usar

Execute o script principal:
```bash
python teste.py
```

- Pressione 'q' para sair do programa
- O programa mostrará sua pose em tempo real com as landmarks desenhadas
- Os ângulos do joelho e quadril serão exibidos no console

## Dependências

- OpenCV (cv2)
- MediaPipe
- NumPy

## Estrutura do Projeto

- `teste.py`: Script principal com a detecção de pose
- `requirements.txt`: Lista de dependências do projeto
- `README.md`: Documentação do projeto

## Como funciona

O projeto utiliza a biblioteca MediaPipe da Google para detectar landmarks da pose humana em tempo real. As principais funcionalidades incluem:

1. **Captura de vídeo**: Utiliza OpenCV para capturar frames da webcam
2. **Detecção de pose**: MediaPipe processa cada frame e identifica pontos-chave do corpo
3. **Cálculo de ângulos**: Função personalizada para calcular ângulos entre três pontos
4. **Visualização**: Desenha as landmarks e conexões sobre a imagem original

## Possíveis melhorias

- Adicionar contador de repetições para exercícios
- Implementar feedback sobre a qualidade do movimento
- Salvar dados de treino em arquivo
- Adicionar mais tipos de exercícios
- Interface gráfica mais avançada

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.