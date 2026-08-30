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

// 场景（OpenCode 线上故障回归）：anthropic 分组里的 OpenAI 格式账号不支持扩展
// 档位，账号管理里给它配置 max_reasoning_effort=high。Claude Code 开 ultracode
// 时，调度到该账号的请求降档为 high 转发（用户侧无感）；调度到未配置上限的账号
// （如支持 ultracode 的账号）时保持原档位。
func TestForwardAsAnthropic_AccountCeilingClampsUltraTierForChatCompletionsUpstream(t *testing.T) {
	for _, tc := range []struct {
		name         string
		ceiling      string
		effort       string
		wantUpstream string
	}{
		{name: "ultracode clamped by account ceiling", ceiling: "high", effort: "ultracode", wantUpstream: "high"},
		{name: "max clamped by account ceiling", ceiling: "high", effort: "max", wantUpstream: "high"},
		{name: "passthrough without account ceiling", ceiling: "", effort: "ultracode", wantUpstream: "xhigh"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gin.SetMode(gin.TestMode)

			body := []byte(`{"model":"claude-opus-5","max_tokens":16,"stream":false,"output_config":{"effort":"` + tc.effort + `"},"messages":[{"role":"user","content":"hello"}]}`)
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			upstream := &httpUpstreamRecorder{resp: &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_acct_ceiling"}},
				Body: io.NopCloser(strings.NewReader(
					`{"id":"chatcmpl_acct","object":"chat.completion","model":"mimo-v2.5","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
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
			if tc.ceiling != "" {
				account.Extra["max_reasoning_effort"] = tc.ceiling
			}

			result, err := svc.ForwardAsAnthropic(context.Background(), c, account, body, "", "")
			require.NoError(t, err)
			require.NotNil(t, result)
			require.Equal(t, "mimo-v2.5", result.UpstreamModel)
			require.Equal(t, tc.wantUpstream, gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
			require.Equal(t, http.StatusOK, rec.Code)
		})
	}
}
