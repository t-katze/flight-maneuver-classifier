"""
Tests for maneuver_acmi_export.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_acmi_export import (
    build_annotation_schedule,
    extract_frame_markers,
    extract_object_first_frames,
    inject_annotation_lines,
    load_labeled_windows,
)


SAMPLE_ACMI = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
#0
1,T=44.1|41.1|5000,Name=F-16C,Type=Air+FixedWing
#1
1,T=44.2|41.2|5100
#2
1,T=44.3|41.3|5200
"""


class TestBuildAnnotationSchedule:
    def test_assigns_to_nearest_frame_from_center(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.2,
                    "aircraft_id": "1",
                    "label": 1,
                    "label_name": "Climb",
                }
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="center")

        assert "1" in schedule
        assert schedule["1"] == [
            "1,ManeuverLabel=Climb,ManeuverLabelId=1,"
            "ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6,"
            "ManeuverRuleBasedLabel=Climb,ManeuverRuleBasedLabelId=1"
        ]

    def test_last_update_wins_for_same_frame_and_object(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "1",
                    "label": 0,
                    "label_name": "Level Flight",
                },
                {
                    "window_start": 0.1,
                    "window_end": 1.1,
                    "aircraft_id": "1",
                    "label": 3,
                    "label_name": "Left Turn",
                },
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="start")

        assert schedule["0"] == [
            "1,ManeuverLabel=Left Turn,ManeuverLabelId=3,"
            "ManeuverWindowStart=0.1,ManeuverWindowEnd=1.1,ManeuverSampleTime=0.1,"
            "ManeuverRuleBasedLabel=Left Turn,ManeuverRuleBasedLabelId=3"
        ]

    def test_skips_unknown_objects_and_respects_first_frame(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        first_frames = extract_object_first_frames(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": -2.0,
                    "window_end": 0.0,
                    "aircraft_id": "1",
                    "label": 0,
                    "label_name": "Level Flight",
                },
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "999",
                    "label": 1,
                    "label_name": "Climb",
                },
            ]
        )

        schedule = build_annotation_schedule(
            frame_markers,
            labels_df,
            time_anchor="start",
            first_frame_by_object=first_frames,
        )

        assert list(schedule.keys()) == ["0"]
        assert schedule["0"] == [
            "1,ManeuverLabel=Level Flight,ManeuverLabelId=0,"
            "ManeuverWindowStart=-2,ManeuverWindowEnd=0,ManeuverSampleTime=-2,"
            "ManeuverRuleBasedLabel=Level Flight,ManeuverRuleBasedLabelId=0"
        ]

    def test_uses_predicted_label_as_primary_and_keeps_rule_based_label(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "1",
                    "label": 1,
                    "label_name": "Climb",
                    "predicted_label": 3,
                    "predicted_label_name": "Left Turn",
                }
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="start")

        assert schedule["0"] == [
            "1,ManeuverLabel=Left Turn,ManeuverLabelId=3,"
            "ManeuverWindowStart=0,ManeuverWindowEnd=1,ManeuverSampleTime=0,"
            "ManeuverRuleBasedLabel=Climb,ManeuverRuleBasedLabelId=1"
        ]


class TestInjectAnnotationLines:
    def test_inserts_updates_immediately_after_frame_marker(self):
        annotated = inject_annotation_lines(
            SAMPLE_ACMI,
            {
                "1": [
                    "1,ManeuverLabel=Climb,ManeuverLabelId=1,"
                    "ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6,"
                    "ManeuverRuleBasedLabel=Climb,ManeuverRuleBasedLabelId=1"
                ]
            },
        )

        expected = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
#0
1,T=44.1|41.1|5000,Name=F-16C,Type=Air+FixedWing
#1
1,ManeuverLabel=Climb,ManeuverLabelId=1,ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6,ManeuverRuleBasedLabel=Climb,ManeuverRuleBasedLabelId=1
1,T=44.2|41.2|5100
#2
1,T=44.3|41.3|5200
"""
        assert annotated == expected


class TestLoadLabeledWindows:
    def test_fills_label_name_from_label_column(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 5.0,
                    "aircraft_id": "302",
                    "label": 4,
                }
            ]
        ).to_csv(csv_path, index=False)

        labels_df = load_labeled_windows(str(csv_path))

        assert labels_df.iloc[0]["label_name"] == "Steady Descent"
        assert labels_df.iloc[0]["rule_based_label"] == 4
        assert labels_df.iloc[0]["rule_based_label_name"] == "Steady Descent"

    def test_fills_predicted_label_name_from_predicted_label_column(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 5.0,
                    "aircraft_id": "302",
                    "label": 4,
                    "label_name": "Steady Descent",
                    "predicted_label": 3,
                }
            ]
        ).to_csv(csv_path, index=False)

        labels_df = load_labeled_windows(str(csv_path))

        assert labels_df.iloc[0]["predicted_label_name"] == "Steady Climb"
