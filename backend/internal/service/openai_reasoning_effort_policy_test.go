package service

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestCanonicalRequestedReasoningEffort(t *testing.T) {
	t.Parallel()

	max := CanonicalRequestedReasoningEffort([]byte(`{"model":"gpt-5.4","reasoning":{"effort":"MAX"}}`), "gpt-5.4")
	require.NotNil(t, max)
	require.Equal(t, "max", *max)

	fromSuffix := CanonicalRequestedReasoningEffort([]byte(`{"model":"gpt-5.4-max"}`), "gpt-5.4-max")
	require.NotNil(t, fromSuffix)
	require.Equal(t, "max", *fromSuffix)

	claude := CanonicalRequestedReasoningEffort([]byte(`{"model":"claude-sonnet-4","output_config":{"effort":"high"}}`))
	require.NotNil(t, claude)
	require.Equal(t, "high", *claude)

	require.Nil(t, CanonicalRequestedReasoningEffort([]byte(`{"model":"gpt-5.4"}`), "gpt-5.4"))
}

func TestRequestedReasoningEffortContext(t *testing.T) {
	t.Parallel()

	require.Nil(t, RequestedReasoningEffortFromContext(context.Background()))
	ctx := WithRequestedReasoningEffort(context.Background(), " max ")
	got := RequestedReasoningEffortFromContext(ctx)
	require.NotNil(t, got)
	require.Equal(t, "max", *got)
}

func TestNormalizeMaxReasoningEffort(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "empty", in: "", want: ""},
		{name: "separator", in: "x-high", want: "xhigh"},
		{name: "max is distinct", in: "max", want: "max"},
		{name: "none is the lowest codex tier", in: "none", want: "none"},
		{name: "claude ultracode alias folds to max", in: "ultracode", want: "max"},
		{name: "ultracode alias case/separator insensitive", in: "Ultra_Code", want: "max"},
		{name: "none is the lowest codex tier", in: "none", want: "none"},
		{name: "invalid", in: "banana", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, NormalizeMaxReasoningEffort(tt.in))
		})
	}
}

func TestNormalizeReasoningEffortMappings(t *testing.T) {
	t.Run("canonicalizes fixed OpenAI values", func(t *testing.T) {
		for _, platform := range []string{PlatformOpenAI, PlatformComposite} {
			got, err := NormalizeReasoningEffortMappings(platform, []ReasoningEffortMapping{
				{From: " MAX ", To: " x-high "},
				{From: "minimal", To: "high"},
			})
			require.NoError(t, err)
			require.Equal(t, []ReasoningEffortMapping{
				{From: "max", To: "xhigh"},
				{From: "minimal", To: "high"},
			}, got)
		}
	})

	t.Run("rejects empty values", func(t *testing.T) {
		_, err := NormalizeReasoningEffortMappings(PlatformOpenAI, []ReasoningEffortMapping{{From: "max"}})
		require.ErrorContains(t, err, "empty or unknown")
	})

	t.Run("rejects duplicate sources case insensitively", func(t *testing.T) {
		_, err := NormalizeReasoningEffortMappings(PlatformOpenAI, []ReasoningEffortMapping{
			{From: "max", To: "xhigh"},
			{From: " MAX ", To: "high"},
		})
		require.ErrorContains(t, err, "duplicate")
	})

	t.Run("rejects mappings for non OpenAI platforms", func(t *testing.T) {
		for _, platform := range []string{PlatformAnthropic, PlatformGemini, PlatformAntigravity, PlatformGrok} {
			_, err := NormalizeReasoningEffortMappings(platform, []ReasoningEffortMapping{{From: "low", To: "high"}})
			require.ErrorContains(t, err, "only supported for platforms \"openai\" and \"composite\"")
		}

		_, err := NormalizeReasoningEffortMappings(PlatformOpenAI, []ReasoningEffortMapping{{From: "none", To: "low"}})
		// none 能被识别为档位，但不允许作为映射源/目标。
		require.ErrorContains(t, err, "not supported")

		_, err = NormalizeReasoningEffortMappings(PlatformOpenAI, []ReasoningEffortMapping{{From: "ultra", To: "high"}})
		require.ErrorContains(t, err, "empty or unknown")
	})
}

func TestNormalizeMaxReasoningEffortForPlatform(t *testing.T) {
	value, err := normalizeMaxReasoningEffortForPlatform(PlatformOpenAI, "max")
	require.NoError(t, err)
	require.Equal(t, "max", value)
	value, err = normalizeMaxReasoningEffortForPlatform(PlatformComposite, "max")
	require.NoError(t, err)
	require.Equal(t, "max", value)

	for _, platform := range []string{PlatformAnthropic, PlatformGemini, PlatformAntigravity, PlatformGrok} {
		_, err = normalizeMaxReasoningEffortForPlatform(platform, "low")
		require.ErrorContains(t, err, "only supported for platforms \"openai\" and \"composite\"")
	}

	_, err = normalizeMaxReasoningEffortForPlatform(PlatformOpenAI, "none")
	require.ErrorContains(t, err, "not supported")
}

func TestOpenAIReasoningEffortPolicyContext(t *testing.T) {
	body := []byte(`{"reasoning":{"effort":"max"}}`)

	unbound, changed := ApplyOpenAIReasoningEffortPolicyFromContext(context.Background(), body)
	require.False(t, changed)
	require.Equal(t, body, unbound)

	mappings := []ReasoningEffortMapping{{From: "max", To: "xhigh"}}
	ctx := WithOpenAIReasoningEffortPolicy(context.Background(), "medium", mappings)
	mappings[0].To = "low"
	got, changed := ApplyOpenAIReasoningEffortPolicyFromContext(ctx, body)
	require.True(t, changed)
	require.Equal(t, "medium", gjson.GetBytes(got, "reasoning.effort").String())
}

func TestApplyOpenAIReasoningEffortPolicy(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		max      string
		mappings []ReasoningEffortMapping
		path     string
		want     string
		changed  bool
	}{
		{name: "nested caps high", body: `{"reasoning":{"effort":"xhigh"}}`, max: "medium", path: "reasoning.effort", want: "medium", changed: true},
		{name: "flat caps high", body: `{"reasoning_effort":"high"}`, max: "low", path: "reasoning_effort", want: "low", changed: true},
		{name: "does not raise omitted", body: `{"model":"gpt-5"}`, max: "low", path: "reasoning_effort", want: "", changed: false},
		{name: "keeps lower value", body: `{"reasoning_effort":"low"}`, max: "high", path: "reasoning_effort", want: "low", changed: false},
		{name: "normalizes request alias", body: `{"reasoning_effort":"x-high"}`, max: "xhigh", path: "reasoning_effort", want: "xhigh", changed: true},
		{name: "caps max below its distinct rank", body: `{"reasoning_effort":"max"}`, max: "xhigh", path: "reasoning_effort", want: "xhigh", changed: true},
		{name: "keeps xhigh below max", body: `{"reasoning_effort":"xhigh"}`, max: "max", path: "reasoning_effort", want: "xhigh", changed: false},
		{name: "ignores stale none ceiling", body: `{"reasoning_effort":"high"}`, max: "none", path: "reasoning_effort", want: "high", changed: false},
		{name: "caps both shapes", body: `{"reasoning":{"effort":"high"},"reasoning_effort":"xhigh"}`, max: "low", path: "reasoning.effort", want: "low", changed: true},
		{name: "maps before cap", body: `{"reasoning":{"effort":"MAX"}}`, max: "medium", mappings: []ReasoningEffortMapping{{From: "max", To: "xhigh"}}, path: "reasoning.effort", want: "medium", changed: true},
		{name: "does not chain mappings", body: `{"reasoning_effort":"max"}`, mappings: []ReasoningEffortMapping{{From: "max", To: "xhigh"}, {From: "xhigh", To: "low"}}, path: "reasoning_effort", want: "xhigh", changed: true},
		{name: "keeps unknown without mapping", body: `{"reasoning_effort":"future"}`, max: "low", path: "reasoning_effort", want: "future", changed: false},
		{name: "keeps non string value", body: `{"reasoning_effort":{"level":"high"}}`, max: "low", path: "reasoning_effort.level", want: "high", changed: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, changed := ApplyOpenAIReasoningEffortPolicy([]byte(tt.body), tt.max, tt.mappings)
			require.Equal(t, tt.changed, changed)
			if tt.path != "" {
				require.Equal(t, tt.want, gjson.GetBytes(got, tt.path).String())
			}
		})
	}
}

func TestClampReasoningEffortForAccount(t *testing.T) {
	accountWithCeiling := func(ceiling string) *Account {
		return &Account{
			Platform: PlatformOpenAI,
			Type:     AccountTypeAPIKey,
			Extra:    map[string]any{"max_reasoning_effort": ceiling},
		}
	}

	tests := []struct {
		name    string
		effort  string
		account *Account
		want    string
	}{
		// OpenCode 场景：账号配 high，ultracode/max/xhigh 全部降为 high。
		{"ultracode clamped to high", "ultracode", accountWithCeiling("high"), "high"},
		{"max clamped to high", "max", accountWithCeiling("high"), "high"},
		{"xhigh clamped to high", "xhigh", accountWithCeiling("high"), "high"},
		{"high unchanged", "high", accountWithCeiling("high"), "high"},
		{"medium unchanged", "medium", accountWithCeiling("high"), "medium"},
		// none 比一切 ceiling 都低，永不抬升。
		{"none never raised", "none", accountWithCeiling("high"), "none"},
		// 配 max 的账号：扩展档位照常透传（支持 ultracode 的账号）。
		{"ultracode passthrough with max ceiling", "ultracode", accountWithCeiling("max"), "ultracode"},
		{"xhigh passthrough with max ceiling", "xhigh", accountWithCeiling("max"), "xhigh"},
		// 未配置 ceiling：完全不影响。
		{"no ceiling passthrough", "ultracode", accountWithCeiling(""), "ultracode"},
		{"nil account passthrough", "max", nil, "max"},
		// 非法 ceiling 值视为未配置。
		{"invalid ceiling ignored", "max", accountWithCeiling("banana"), "max"},
		// 未认识的未来档位也不允许越过 ceiling。
		{"unknown tier clamped", "hyper-max-plus", accountWithCeiling("high"), "high"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, clampReasoningEffortForAccount(tt.effort, tt.account))
		})
	}
}

func TestAccountGetMaxReasoningEffort(t *testing.T) {
	require.Equal(t, "", (*Account)(nil).GetMaxReasoningEffort())
	require.Equal(t, "", (&Account{Platform: PlatformAnthropic}).GetMaxReasoningEffort(), "非 openai 账号不生效")
	require.Equal(t, "high", (&Account{Platform: PlatformOpenAI, Extra: map[string]any{"max_reasoning_effort": "high"}}).GetMaxReasoningEffort())
	require.Equal(t, "max", (&Account{Platform: PlatformOpenAI, Extra: map[string]any{"max_reasoning_effort": "ULTRA_CODE"}}).GetMaxReasoningEffort())
	require.Equal(t, "", (&Account{Platform: PlatformOpenAI, Extra: map[string]any{"max_reasoning_effort": "banana"}}).GetMaxReasoningEffort(), "非法值视为未配置")
	require.Equal(t, "", (&Account{Platform: PlatformOpenAI}).GetMaxReasoningEffort())
}
