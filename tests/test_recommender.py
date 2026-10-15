import pytest

from klike.data import load_clean
from klike.recommender import recommend

BASIC = {
    "platform": "Meta", "category": "E-commerce", "objective": "traffic", "target_audience_age": "25-34",
    "is_retargeting": False, "video_duration_s": 20, "format": "vertical", "has_subtitle": False,
    "has_cta": False, "has_hook": False, "has_face": False, "text_density": "high", "music_voice_ratio": 0.4,
}


@pytest.fixture(scope="module")
def df():
    return load_clean()


def test_weak_creative_gets_at_least_three_recommendations():
    assert len(recommend(BASIC)["recommendations"]) >= 3


def test_sorted_by_priority_with_hypotheses_last():
    recs = recommend(BASIC)["recommendations"]
    firm = [r for r in recs if not r["hypothesis"]]
    assert [r["priority"] for r in firm] == sorted((r["priority"] for r in firm), reverse=True)
    assert recs[: len(firm)] == firm


def test_never_recommends_what_the_campaign_already_has(df):
    for _, row in df.sample(100, random_state=0).iterrows():
        for r in recommend(row)["recommendations"]:
            attr, current = r["change"]["attribute"], r["change"]["from"]
            assert current != r["change"]["to"], (row["campaign_id"], attr)
            if attr.startswith("has_"):
                assert current is not True


def test_no_horizontal_on_tiktok(df):
    for _, row in df[df["platform"] == "TikTok"].iterrows():
        assert all(r["key"] != "format_horizontal" for r in recommend(row)["recommendations"])


def test_combined_plan_is_at_least_the_best_single_action():
    result = recommend(BASIC)
    best = max(r["score_gain"] for r in result["recommendations"])
    assert result["combined_plan"]["score_gain"] >= best
    assert result["combined_plan"]["score_after"] <= 100
