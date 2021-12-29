# Aprendizado-de-Maquina

Esse repositório contém os exercícios aplicados pelo professor Rodolfo Miranda Pereira na matéria de Aprendizagem de Máquina da IFPR - Campus Pinhais.

## Banco de dados de fim do jogo da velha

**Link :** https://www.openml.org/d/50

**Descrição :** Banco de dados composto por um conjunto completo de possibilidades
de configurações de tabuleiro, do qual o jogador X pode ser considerado vitorioso.
Isso ocorre quando o jogador X tem a possibilidade dentre as 8 de alinhar 3
símbolos iguais.

**Número de instâncias :** 958 instâncias.

**Classe :** positivo e negativo (positive e negative).

**Total das Classes :** 626 positivas e 332 negativas

**Número de atributos :** 10 atributos.

**Atributos :**
  1. top-left-square: {x,o,b}
  2. top-middle-square: {x,o,b}
  3. top-right-square: {x,o,b}
  4. middle-left-square: {x,o,b}
  5. middle-middle-square: {x,o,b}
  6. middle-right-square: {x,o,b}
  7. bottom-left-square: {x,o,b}
  8. bottom-middle-square: {x,o,b}
  9. bottom-right-square: {x,o,b}
  10. Class: {positive,negative}

Os atributos de 1 a 9 podem assumir um dos seguintes resultados :
  * x para os quadrados preenchidos pelo jogador X
  * o para os quadrados preenchidos pelo jogador O
  * b para os quadrados não preenchidos
  * Enquanto a o atributo 10, pode assumir positivo e negativo.

**Autor:** David W. Aha - [UCI](http://archive.ics.uci.edu/ml/citation_policy.html)
