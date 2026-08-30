//go:build unit

package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// 场景（OpenCode 线上故障回归）：Claude Code 开 ultracode/max 档 → /v1/messages →
// anthropic 分组桥接到 OpenAI 格式账号（模型映射 claude-opus-5 -> mimo-v2.5），
// 上游不认识扩展档位。分组配置 max_reasoning_effort=high 后（入口绑定策略，
// 桥接处应用），ultracode/max 一律降为 high 转发，而不是透传导致上游 500。
func TestForwardAsAnthropic_CapsUltraTierByGroupPolicyForChatCompletionsUpstream(t *testing.T) {
	for _, tc := range []struct{ effort string }{{"ultracode"}, {"max"}, {"xhigh"}} {
		t.Run(tc.effort, func(t *testing.T) {
			gin.SetMode(gin.TestMode)

			body := []byte(`{"model":"claude-opus-5","max_tokens":16,"stream":false,"output_config":{"effort":"` + tc.effort + `"},"messages":[{"role":"user","content":"hello"}]}`)
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")
			// 入口 bindOpenAIReasoningEffortPolicyForMessagesRequest 绑定后的 ctx 状态。
			ctx := WithOpenAIReasoningEffortPolicy(context.Background(), "high", nil)

			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_ultra_cap"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"chatcmpl_cap","object":"chat.completion","model":"mimo-v2.5","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
				)),
			}}
			svc := &OpenAIGatewayService{
				cfg:          rawChatCompletionsTestConfig(),
				httpUpstream: upstream,
			}
			account := forceChatResponsesFallbackAccount()
			account.Credentials["model_mapping"] = map[string]any{
				"claude-opus-5": "mimo-v2.5",
			}

			result, err := svc.ForwardAsAnthropic(ctx, c, account, body, "", "")
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, "mimo-v2.5", result.UpstreamModel)
			require.Equal(t, "high", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
			require.NotNil(t, result.ReasoningEffort)
			require.Equal(t, "high", *result.ReasoningEffort)
			require.Equal(t, http.StatusOK, rec.Code)
		})
	}
}

// 对照：分组未配置 ceiling 时，max 按既有语义归一为 xhigh 透传。
func TestForwardAsAnthropic_PassthroughUltraTierWithoutGroupPolicy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"claude-opus-5","max_tokens":16,"stream":false,"output_config":{"effort":"max"},"messages":[{"role":"user","content":"hello"}]}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_ultra_pass"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_pass","object":"chat.completion","model":"mimo-v2.5","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}
	account := forceChatResponsesFallbackAccount()
	account.Credentials["model_mapping"] = map[string]any{
		"claude-opus-5": "mimo-v2.5",
	}

	_, err := svc.ForwardAsAnthropic(context.Background(), c, account, body, "", "")
	require.NoError(t, err)
	require.Equal(t, "xhigh", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
	require.Equal(t, http.StatusOK, rec.Code)
}
