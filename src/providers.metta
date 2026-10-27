(= (llmProviderStart $provider)
   (py-call (providers.llmProviderStart $provider)))

(= (llmProviderChat $prompt $max_tokens $reasoning_mode)
   (py-call (providers.llmProviderChat $prompt $max_tokens $reasoning_mode)))

;Notice text for the user if the provider reply is the output limit marker, otherwise ""
(= (llmLimitNotice $response)
   (py-call (lib_llm_ext.llmLimitNotice $response)))
