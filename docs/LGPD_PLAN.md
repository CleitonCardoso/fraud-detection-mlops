# Plano de Conformidade LGPD

> Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais

---

## Mapeamento de Dados Pessoais

| Dado | Presente no sistema? | Justificativa |
|---|---|---|
| Nome | Não | Dataset usa PCA — sem campos identificáveis |
| CPF / RG | Não | Idem |
| Email | Não | Idem |
| Número de cartão | Não | Idem — features são transformadas |
| Valor de transação | Sim (Amount) | Dado financeiro, não diretamente pessoal |
| Timestamp | Sim (Time) | Relativo, sem referência absoluta de data |
| IP / localização | Não | Não presentes no dataset |

**Conclusão**: O dataset não contém dados pessoais identificáveis. As features V1–V28 são componentes de Análise de Componentes Principais (PCA) aplicada aos dados originais — a transformação é irreversível sem a matriz de transformação original, que não está disponível.

---

## Base Legal

| Processamento | Base Legal (Art. 7º LGPD) |
|---|---|
| Treinamento do modelo | Art. 7º, IX — legítimo interesse em prevenção à fraude |
| Inferência em produção | Art. 7º, II — execução de contrato (proteção do titular) |
| Logs de auditoria | Art. 7º, IX — legítimo interesse em segurança |

---

## Direitos dos Titulares

| Direito | Como atendemos |
|---|---|
| Acesso (Art. 18, I) | Dataset público — sem dados de titulares identificáveis |
| Correção (Art. 18, III) | Não aplicável — dados anonimizados |
| Exclusão (Art. 18, VI) | Não aplicável — dados anonimizados |
| Portabilidade (Art. 18, V) | Não aplicável |
| Informação (Art. 18, VII) | Este documento e o System Card descrevem o processamento |

---

## Medidas Técnicas de Proteção

- **Presidio**: detecta e anonimiza PII em inputs e outputs do agente em tempo real
- **Guardrails**: bloqueia inputs com padrões suspeitos antes de processar
- **DVC**: dados versionados mas nunca commitados no Git público
- **Logs**: estruturados sem dados de transação — apenas métricas e timestamps
- **Acesso**: credenciais via `.env` local, nunca hardcoded ou commitadas

---

## Responsável pelo Tratamento (DPO)

Ver tag `owner` no MLflow Registry para o responsável técnico da versão em produção.
