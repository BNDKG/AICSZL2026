from pathlib import Path

import pandas as pd

from aicszl.blends.runner import BlendInput, BlendJob, blend_predictions


def test_blend_predictions_writes_weighted_mean_daily_rank_pkl(tmp_path: Path):
    first_path = tmp_path / "pred_a.pkl"
    second_path = tmp_path / "pred_b.pkl"
    pd.DataFrame(
        [
            _prediction("000001.SZ", 20200102, 1.0),
            _prediction("000002.SZ", 20200102, 3.0),
            _prediction("000001.SZ", 20200103, 2.0),
        ]
    ).to_pickle(first_path)
    pd.DataFrame(
        [
            _prediction("000001.SZ", 20200102, 5.0),
            _prediction("000002.SZ", 20200102, 2.0),
            _prediction("000001.SZ", 20200103, 4.0),
        ]
    ).to_pickle(second_path)

    artifact = blend_predictions(
        BlendJob(
            name="blend_rank5_v1",
            inputs=[
                BlendInput(prediction_id="pred_a", path=first_path, weight=1.0),
                BlendInput(prediction_id="pred_b", path=second_path, weight=0.5),
            ],
        ),
        tmp_path / "artifacts" / "blends",
    )

    assert artifact.blend_path.exists()
    result = pd.read_pickle(artifact.blend_path)
    assert result.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200102,
            "score_raw_blend": 2.3333333333333335,
            "score_rank_blend": 0.5,
            "input_prediction_ids": "pred_a,pred_b",
            "blend_job_id": "blend_rank5_v1",
        },
        {
            "ts_code": "000002.SZ",
            "trade_date": 20200102,
            "score_raw_blend": 2.6666666666666665,
            "score_rank_blend": 1.0,
            "input_prediction_ids": "pred_a,pred_b",
            "blend_job_id": "blend_rank5_v1",
        },
        {
            "ts_code": "000001.SZ",
            "trade_date": 20200103,
            "score_raw_blend": 2.6666666666666665,
            "score_rank_blend": 1.0,
            "input_prediction_ids": "pred_a,pred_b",
            "blend_job_id": "blend_rank5_v1",
        },
    ]


def _prediction(ts_code: str, trade_date: int, score_raw: float) -> dict[str, object]:
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "score_raw": score_raw,
        "score_rank": 1.0,
        "model_artifact_id": "model",
        "train_job_id": "job",
        "x_group": "base_v1",
        "y_name": "target.ret_5d_rank_pct.v1",
    }
