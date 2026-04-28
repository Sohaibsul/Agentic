import os

from agents.data_agent import DataAgent
from agents.decision_agent import DecisionAgent
from agents.imbalance_correction_agent import ImbalanceCorrectionAgent
from agents.classification_agent import ClassificationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.xai_agent import XAIAgent
from agents.report_agent import ReportAgent
from agents.crewai_report_agent import run_crewai_report

from utils.visualization import (
    save_confusion_matrix,
    save_class_distribution_chart,
    save_results_table_csv
)


def main():
    os.makedirs("outputs", exist_ok=True)

    print("\nStarting Agentic Eye Disease Classification System...\n")

    data_agent = DataAgent("test_images")
    decision_agent = DecisionAgent()
    imbalance_agent = ImbalanceCorrectionAgent()
    classification_agent = ClassificationAgent()
    evaluation_agent = EvaluationAgent()
    xai_agent = XAIAgent()
    report_agent = ReportAgent()

    # ---------------- DATA AUDIT ----------------
    audit = data_agent.audit()
    print("DATA AUDIT:", audit)

    # ---------------- DECISION ----------------
    balance_decision = decision_agent.decide_balance_strategy(audit)
    print("BALANCE DECISION:", balance_decision)

    # ---------------- IMBALANCE ----------------
    imbalance_suggestion = imbalance_agent.suggest(audit)
    print("IMBALANCE SUGGESTION:", imbalance_suggestion)

    # ---------------- AGENTIC DECISION EVALUATION ----------------
    agentic_evaluation = evaluation_agent.evaluate_agentic_decision(
        audit_result=audit,
        balance_decision=balance_decision
    )

    # ---------------- CLASSIFICATION ----------------
    predictions = []

    for image_path in audit["images"]:
        pred = classification_agent.classify(image_path)

        true_label = data_agent.infer_label_from_path(image_path)
        pred["true_label"] = true_label

        reliability = decision_agent.decide_prediction_reliability(
            pred["confidence"]
        )
        pred["reliability"] = reliability

        if pred["confidence"] < 0.90:
            xai_result = xai_agent.explain(
                image_path=image_path,
                model=classification_agent.model,
                device=classification_agent.device,
                id2label=classification_agent.id2label
            )
            pred["gradcam_path"] = xai_result["gradcam_path"]
        else:
            pred["gradcam_path"] = None

        predictions.append(pred)

        print(
            f"{os.path.basename(image_path)} → "
            f"True: {true_label} | "
            f"Predicted: {pred['prediction']} "
            f"({pred['confidence']:.2f}) | {reliability}"
        )

    # ---------------- EVALUATION ----------------
    evaluation = evaluation_agent.summarize_predictions(predictions)

    performance_metrics = evaluation_agent.evaluate_with_ground_truth(
        predictions
    )

    # ---------------- SAVE CHARTS AND TABLES ----------------
    confusion_matrix_path = save_confusion_matrix(
        confusion_matrix=performance_metrics["confusion_matrix"],
        labels=performance_metrics["labels"],
        output_path="outputs/confusion_matrix.png"
    )

    class_distribution_path = save_class_distribution_chart(
        class_counts=audit["class_counts"],
        output_path="outputs/class_distribution.png"
    )

    results_table_path = save_results_table_csv(
        performance_metrics=performance_metrics,
        agentic_evaluation=agentic_evaluation,
        output_path="outputs/results_table.csv"
    )

    print("\nSAVED VISUAL OUTPUTS")
    print("Confusion matrix:", confusion_matrix_path)
    print("Class distribution chart:", class_distribution_path)
    print("Results table:", results_table_path)

    print("\nPERFORMANCE METRICS")
    print(f"Accuracy: {performance_metrics['accuracy']:.2f}")
    print(f"Precision: {performance_metrics['precision']:.2f}")
    print(f"Recall: {performance_metrics['recall']:.2f}")
    print(f"F1-score: {performance_metrics['f1_score']:.2f}")
    print("\nClassification Report:")
    print(performance_metrics["classification_report"])

    print("\nAGENTIC EVALUATION")
    print(agentic_evaluation)

    # ---------------- REPORT ----------------
    final_report = report_agent.save_report(
        audit=audit,
        balance_decision=balance_decision,
        imbalance_suggestion=imbalance_suggestion,
        predictions=predictions,
        evaluation=evaluation,
        performance_metrics=performance_metrics,
        agentic_evaluation=agentic_evaluation
    )

    # ---------------- CREWAI ----------------
    crewai_summary = run_crewai_report(final_report)

    with open("outputs/crewai_summary.txt", "w", encoding="utf-8") as f:
        f.write(crewai_summary)

    print("\nCrewAI Summary:")
    print(crewai_summary)

    print("\n" + final_report)
    print("\nDone. Check outputs/ folder.")


if __name__ == "__main__":
    main()