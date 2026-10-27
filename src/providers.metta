(= (llmProviderStart $provider)
   (py-call (providers.llmProviderStart $provider)))

(= (llmProviderChat $prompt $max_tokens $reasoning_mode)
   (py-call (providers.llmProviderChat $prompt $max_tokens $reasoning_mode)))


(= (llmProviderRespond $prompt $max_tokens $reasoning_mode)
   (py-call (providers.llmProviderRespond $prompt $max_tokens $reasoning_mode)))

(= (response-field $response $name)
   (py-call (providers.response_field $response $name)))
