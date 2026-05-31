import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, make_scorer, fbeta_score
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from imblearn.pipeline import Pipeline
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    from sklearn.pipeline import Pipeline
    IMBLEARN_AVAILABLE = False
    print("⚠️ imbalanced-learn no instalado. Instala con: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Instala con: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Instala con: pip install tensorflow")
from sklearn.ensemble import ExtraTreesClassifier
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP no disponible. Instala con: pip install shap")


class PrecipitationEventModelPipeline:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names=None,
        threshold_metric="f2",
        beta=1.5,
        random_state=42
    ):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()

        self.feature_names = (
            feature_names
            if feature_names is not None
            else X_train.columns.tolist()
        )

        self.threshold_metric = threshold_metric
        self.beta = beta
        self.random_state = random_state

        self.pipelines = {}
        self.models = {}
        self.probas = {}
        self.predictions = {}
        self.thresholds = {}
        self.metrics = {}
        self.confusion_matrices = {}
        self.shap_values = {}

        neg = np.sum(self.y_train == 0)
        pos = np.sum(self.y_train == 1)

        self.ratio_real = float(neg / pos) if pos > 0 else 1.0

        # Peso suavizado para no volver loco al modelo
        self.scale_pos_weight = np.sqrt(self.ratio_real)

        # Bias inicial para redes neuronales
        self.initial_bias = np.log(pos / neg) if pos > 0 and neg > 0 else 0.0

        print("📊 Distribución train:")
        print(pd.Series(self.y_train).value_counts(normalize=True).rename("proporción"))
        print(f"\n⚖️ Ratio real negativos/positivos: {self.ratio_real:.2f}")
        print(f"⚖️ Peso suavizado positivos: {self.scale_pos_weight:.2f}")

    # =========================================================
    # MÉTRICAS
    # =========================================================

    def _find_best_threshold(self, y_true, proba):
        precision, recall, thresholds = precision_recall_curve(y_true, proba)

        # precision y recall tienen un elemento más que thresholds
        precision = precision[:-1]
        recall = recall[:-1]

        if len(thresholds) == 0:
            return 0.5

        if self.threshold_metric == "f2":
            beta2 = self.beta ** 2
            scores = (1 + beta2) * precision * recall / (
                beta2 * precision + recall + 1e-10
            )

        elif self.threshold_metric == "custom":
            # Métrica personalizada:
            # penaliza mucho falsos negativos,
            # pero también controla falsos positivos.
            scores = []

            for thr in thresholds:
                pred = (proba >= thr).astype(int)
                cm = confusion_matrix(y_true, pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                score = tp - 3.0 * fn - 0.5 * fp
                scores.append(score)

            scores = np.array(scores)

        else:
            scores = fbeta_score(
                y_true,
                (proba >= 0.5).astype(int),
                beta=self.beta,
                zero_division=0
            )
            return 0.5

        best_idx = int(np.argmax(scores))
        best_threshold = float(thresholds[best_idx])

        return best_threshold

    def _evaluate_one_model(self, model_name):
        y_pred = self.predictions[model_name]
        y_proba = self.probas[model_name]

        cm = confusion_matrix(self.y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        self.confusion_matrices[model_name] = cm

        metrics = {
            "Threshold": self.thresholds[model_name],
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision Extremo": precision_score(
                self.y_test, y_pred, pos_label=1, zero_division=0
            ),
            "Recall Extremo": recall_score(
                self.y_test, y_pred, pos_label=1, zero_division=0
            ),
            "F1 Extremo": f1_score(
                self.y_test, y_pred, pos_label=1, zero_division=0
            ),
            "F2 Extremo": fbeta_score(
                self.y_test, y_pred, beta=2.0, pos_label=1, zero_division=0
            ),
            "F1 Macro": f1_score(
                self.y_test, y_pred, average="macro", zero_division=0
            ),
            "ROC-AUC": roc_auc_score(self.y_test, y_proba),
            "PR-AUC": average_precision_score(self.y_test, y_proba),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }

        self.metrics[model_name] = metrics

    # =========================================================
    # MODELOS
    # =========================================================

    def _create_models(self):
        models = {}

        # =====================================================
        # MODELOS NORMALES CON PESOS
        # =====================================================

        models["Logistic Regression"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=self.random_state
            ))
        ])

        models["Random Forest"] = Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        models["HistGradientBoosting"] = Pipeline([
            ("model", HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=self.random_state
            ))
        ])

        models["SVM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=self.random_state
            ))
        ])

        models["Extra Trees"] = Pipeline([
            ("model", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=16,
                min_samples_leaf=3,
                min_samples_split=5,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        models["Extra Trees Under"] = Pipeline([
            ("under", RandomUnderSampler(
                sampling_strategy=0.20,
                random_state=self.random_state
            )),
            ("model", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=16,
                min_samples_leaf=3,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = Pipeline([
                ("model", xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=self.scale_pos_weight,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

        # =====================================================
        # MODELOS CON UNDERSAMPLING SUAVE
        # =====================================================
        # sampling_strategy=0.20 significa:
        # positivos / negativos = 0.20
        # o sea, aprox 1 extremo por cada 5 no extremos.
        #
        # Aquí NO ponemos class_weight='balanced'
        # para no empujar doblemente hacia la clase extrema.
        # =====================================================

        if IMBLEARN_AVAILABLE:

            models["Logistic Regression Under"] = Pipeline([
                ("under", RandomUnderSampler(
                    sampling_strategy=0.20,
                    random_state=self.random_state
                )),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=self.random_state
                ))
            ])

            models["Logistic Regression Under 010"] = Pipeline([
                ("under", RandomUnderSampler(
                    sampling_strategy=0.10,
                    random_state=self.random_state
                )),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=self.random_state
                ))
            ])

            models["Logistic Regression NoWeight"] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=self.random_state
                ))
            ])

            models["Random Forest Under"] = Pipeline([
                ("under", RandomUnderSampler(
                    sampling_strategy=0.20,
                    random_state=self.random_state
                )),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

            models["Random Forest NoWeight"] = Pipeline([
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

            models["Random Forest Under 010"] = Pipeline([
                ("under", RandomUnderSampler(
                    sampling_strategy=0.10,
                    random_state=self.random_state
                )),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

            models["SVM Under"] = Pipeline([
                ("under", RandomUnderSampler(
                    sampling_strategy=0.20,
                    random_state=self.random_state
                )),
                ("scaler", StandardScaler()),
                ("model", SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=self.random_state
                ))
            ])

            models["Random Forest SMOTE"] = Pipeline([
                ("smote", SMOTE(
                    sampling_strategy=0.20,
                    random_state=self.random_state,
                    k_neighbors=5
                )),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

            models["Extra Trees SMOTE"] = Pipeline([
                ("smote", SMOTE(
                    sampling_strategy=0.20,
                    random_state=self.random_state,
                    k_neighbors=5
                )),
                ("model", ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=16,
                    min_samples_leaf=3,
                    min_samples_split=5,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])

            models["HistGradientBoosting SMOTE"] = Pipeline([
                ("smote", SMOTE(
                    sampling_strategy=0.20,
                    random_state=self.random_state,
                    k_neighbors=5
                )),
                ("model", HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=0.1,
                    random_state=self.random_state
                ))
            ])

            models["Logistic Regression SMOTE"] = Pipeline([
                ("smote", SMOTE(
                    sampling_strategy=0.20,
                    random_state=self.random_state,
                    k_neighbors=5
                )),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=self.random_state
                ))
            ])


            if XGBOOST_AVAILABLE:
                models["XGBoost Under"] = Pipeline([
                    ("under", RandomUnderSampler(
                        sampling_strategy=0.20,
                        random_state=self.random_state
                    )),
                    ("model", xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        random_state=self.random_state,
                        n_jobs=-1
                    ))
                ])

        # =====================================================
        # RED NEURONAL TABULAR
        # =====================================================

        if TENSORFLOW_AVAILABLE:
            models["Neural Network"] = self._create_tabular_neural_network()

        return models
    def _create_tabular_neural_network(self):
        input_dim = self.X_train.shape[1]

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),

            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),

            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),

            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(
                1,
                activation="sigmoid",
                bias_initializer=keras.initializers.Constant(self.initial_bias)
            )
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(name="roc_auc", curve="ROC"),
                keras.metrics.AUC(name="pr_auc", curve="PR"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.Precision(name="precision")
            ]
        )

        return model

    # =========================================================
    # ENTRENAMIENTO
    # =========================================================

    def train(self, models=None, optimize=True, verbose=True):
        available_models = self._create_models()

        if models is None:
            models_to_train = available_models
        else:
            if isinstance(models, str):
                models = [models]

            models_to_train = {
                name: available_models[name]
                for name in models
                if name in available_models
            }

        print("\n🚀 Entrenando modelos...\n")

        f2_scorer = make_scorer(
            fbeta_score,
            beta=self.beta,
            pos_label=1,
            zero_division=0
        )

        tscv = TimeSeriesSplit(n_splits=3)

        param_grids = {
            "Logistic Regression": {
                "model__C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
            },
            "Random Forest": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [8, 12, 16, 20, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10]
            },
            "SVM": {
                "model__C": [0.1, 0.5, 1, 2, 5, 10],
                "model__gamma": ["scale", "auto", 0.01, 0.05, 0.1]
            }
        }

        if XGBOOST_AVAILABLE:
            param_grids["XGBoost"] = {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__scale_pos_weight": [
                    self.scale_pos_weight,
                    self.ratio_real,
                    max(1.0, self.scale_pos_weight * 1.5)
                ]
            }

        for model_name, model in models_to_train.items():

            if verbose:
                print(f"▶️ Entrenando {model_name}...")

            try:
                if model_name == "Neural Network":
                    self._train_neural_network(model_name, model, verbose=verbose)

                else:
                    if optimize and model_name in param_grids:
                        search = RandomizedSearchCV(
                            estimator=model,
                            param_distributions=param_grids[model_name],
                            n_iter=15,
                            scoring=f2_scorer,
                            cv=tscv,
                            random_state=self.random_state,
                            n_jobs=-1,
                            verbose=0
                        )

                        search.fit(self.X_train, self.y_train)
                        best_model = search.best_estimator_

                        if verbose:
                            print(f"   ✅ Mejor F2 CV: {search.best_score_:.4f}")
                            print(f"   ✅ Params: {search.best_params_}")

                        self.pipelines[model_name] = best_model
                        self.models[model_name] = best_model.named_steps["model"]

                    else:
                        model.fit(self.X_train, self.y_train)
                        self.pipelines[model_name] = model
                        self.models[model_name] = model.named_steps["model"]

                    self._predict_with_threshold_optimization(model_name)

                if verbose:
                    print(f"   ✅ {model_name} completado\n")

            except Exception as e:
                print(f"   ❌ Error en {model_name}: {e}\n")

        self.evaluate()

    def _train_neural_network(self, model_name, model, verbose=True):
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        y_train_np = np.asarray(self.y_train)

        neg = np.sum(y_train_np == 0)
        pos = np.sum(y_train_np == 1)

        weight_for_0 = len(y_train_np) / (2.0 * neg)
        weight_for_1 = len(y_train_np) / (2.0 * pos)

        # Suavizamos el peso de la clase positiva
        weight_for_1 = np.sqrt(weight_for_1 * self.ratio_real)

        class_weight = {
            0: weight_for_0,
            1: weight_for_1
        }

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=12,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]

        model.fit(
            X_train_scaled,
            y_train_np,
            validation_split=0.2,
            epochs=80,
            batch_size=512,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )

        self.pipelines[model_name] = {
            "scaler": scaler,
            "model": model
        }

        self.models[model_name] = model

        proba_train = model.predict(X_train_scaled, verbose=0).ravel()
        threshold = self._find_best_threshold(self.y_train, proba_train)

        proba_test = model.predict(X_test_scaled, verbose=0).ravel()
        y_pred = (proba_test >= threshold).astype(int)

        self.probas[model_name] = proba_test
        self.predictions[model_name] = y_pred
        self.thresholds[model_name] = threshold

    def _predict_with_threshold_optimization(self, model_name):
        pipeline = self.pipelines[model_name]

        proba_train = pipeline.predict_proba(self.X_train)[:, 1]
        threshold = self._find_best_threshold(self.y_train, proba_train)

        proba_test = pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = (proba_test >= threshold).astype(int)

        self.probas[model_name] = proba_test
        self.predictions[model_name] = y_pred
        self.thresholds[model_name] = threshold

    def evaluate(self):
        for model_name in self.predictions.keys():
            self._evaluate_one_model(model_name)

    def get_summary_df(self):
        return pd.DataFrame(self.metrics).T.sort_values(
            by="F2 Extremo",
            ascending=False
        ).round(4)

    # =========================================================
    # PLOTS
    # =========================================================

    def plot_confusion_matrices(self, figsize=None):
        n_models = len(self.confusion_matrices)

        if n_models == 0:
            print("⚠️ No hay matrices de confusión.")
            return

        cols = 2 if n_models > 1 else 1
        rows = int(np.ceil(n_models / cols))

        if figsize is None:
            figsize = (6 * cols, 5 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, cm) in enumerate(self.confusion_matrices.items()):
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                xticklabels=["No extremo", "Extremo"],
                yticklabels=["No extremo", "Extremo"]
            )

            m = self.metrics[model_name]

            axes[idx].set_title(
                f"{model_name}\n"
                f"Recall: {m['Recall Extremo']:.3f} | "
                f"Precision: {m['Precision Extremo']:.3f} | "
                f"F2: {m['F2 Extremo']:.3f}"
            )

            axes[idx].set_xlabel("Predicción")
            axes[idx].set_ylabel("Real")

        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curves(self, figsize=(10, 7)):
        plt.figure(figsize=figsize)

        for model_name, proba in self.probas.items():
            precision, recall, _ = precision_recall_curve(self.y_test, proba)
            ap = average_precision_score(self.y_test, proba)

            plt.plot(
                recall,
                precision,
                label=f"{model_name} | PR-AUC={ap:.3f}"
            )

        plt.xlabel("Recall clase extrema")
        plt.ylabel("Precision clase extrema")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, top_n=20, figsize=(10, 8)):
        importances = {}

        for model_name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = pd.Series(
                    model.feature_importances_,
                    index=self.feature_names
                ).sort_values(ascending=False).head(top_n)

                importances[model_name] = imp

            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_).ravel()
                imp = pd.Series(
                    coef,
                    index=self.feature_names
                ).sort_values(ascending=False).head(top_n)

                importances[model_name] = imp

        if not importances:
            print("⚠️ No hay feature importance disponible.")
            return

        for model_name, imp in importances.items():
            plt.figure(figsize=figsize)
            imp.sort_values().plot(kind="barh")
            plt.title(f"Feature importance - {model_name}")
            plt.xlabel("Importancia")
            plt.tight_layout()
            plt.show()

    # =========================================================
    # SHAP ROBUSTO
    # =========================================================

    # =========================================================
    # EXPLICABILIDAD PARA HISTGRADIENTBOOSTING
    # =========================================================

    def compute_shap_importance(
        self,
        model_name,
        max_samples=300,
        background_size=100,
        random_state=42
    ):
        """
        Calcula SHAP values para un modelo entrenado y los guarda en self.shap_values.
        Funciona para:
        - Random Forest
        - Extra Trees
        - HistGradientBoosting
        - Logistic Regression
        - SVM
        - Neural Network
        """

        if not SHAP_AVAILABLE:
            print("⚠️ SHAP no está disponible. Instala con: pip install shap")
            return None

        if model_name not in self.pipelines:
            print(f"⚠️ {model_name} no está entrenado.")
            return None

        rng = np.random.default_rng(random_state)

        sample_size = min(max_samples, len(self.X_test))
        background_size = min(background_size, len(self.X_train))

        sample_idx = rng.choice(
            len(self.X_test),
            size=sample_size,
            replace=False
        )

        bg_idx = rng.choice(
            len(self.X_train),
            size=background_size,
            replace=False
        )

        X_sample = self.X_test.iloc[sample_idx].copy()
        X_background = self.X_train.iloc[bg_idx].copy()

        pipeline = self.pipelines[model_name]

        print(f"🎯 Calculando SHAP para {model_name}...")
        print(f"   Muestra explicada: {sample_size}")
        print(f"   Background: {background_size}")

        try:
            # =====================================================
            # Neural Network
            # =====================================================
            if model_name == "Neural Network":
                scaler = pipeline["scaler"]
                nn_model = pipeline["model"]

                X_sample_proc = scaler.transform(X_sample)
                X_background_proc = scaler.transform(X_background)

                explainer = shap.Explainer(
                    nn_model.predict,
                    X_background_proc
                )

                shap_values = explainer(X_sample_proc)

                shap_values = shap.Explanation(
                    values=shap_values.values,
                    base_values=shap_values.base_values,
                    data=X_sample.values,
                    feature_names=self.feature_names
                )

            # =====================================================
            # Modelos de árboles compatibles directamente
            # =====================================================
            elif model_name in [
                "Random Forest",
                "Random Forest Under",
                "Random Forest NoWeight",
                "Random Forest Under 010",
                "Random Forest SMOTE",
                "Extra Trees",
                "Extra Trees Under",
                "Extra Trees SMOTE",
                "XGBoost",
                "XGBoost Under"
            ]:
                model = self.models[model_name]

                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(X_sample)

                if isinstance(shap_values_raw, list):
                    shap_values_raw = shap_values_raw[1]

                # Algunos SHAP devuelven shape (n, features, clases)
                if hasattr(shap_values_raw, "ndim") and shap_values_raw.ndim == 3:
                    shap_values_raw = shap_values_raw[:, :, 1]

                shap_values = shap.Explanation(
                    values=shap_values_raw,
                    data=X_sample.values,
                    feature_names=self.feature_names
                )

            # =====================================================
            # HistGradientBoosting y modelos genéricos
            # =====================================================
            else:
                def predict_positive(data):
                    data_df = pd.DataFrame(
                        data,
                        columns=self.feature_names
                    )
                    return pipeline.predict_proba(data_df)[:, 1]

                explainer = shap.Explainer(
                    predict_positive,
                    X_background
                )

                shap_values = explainer(X_sample)

            self.shap_values[model_name] = shap_values

            print(f"✅ SHAP completado para {model_name}")

            return shap_values

        except Exception as e:
            print(f"❌ Error calculando SHAP para {model_name}: {e}")
            return None
        
    def get_shap_importance_df(
        self,
        model_name,
        top_n=30
    ):
        """
        Devuelve una tabla con la importancia SHAP media absoluta.
        """

        if model_name not in self.shap_values:
            print(f"⚠️ No hay SHAP calculado para {model_name}.")
            return None

        shap_values = self.shap_values[model_name]

        values = shap_values.values

        # Por si vienen dimensiones extra
        if values.ndim == 3:
            values = values[:, :, 1]

        mean_abs_shap = np.abs(values).mean(axis=0)

        df_shap = (
            pd.DataFrame({
                "variable": self.feature_names,
                "shap_importance": mean_abs_shap
            })
            .sort_values("shap_importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        return df_shap
    def plot_shap_importance_bar(
        self,
        model_name,
        max_display=20
    ):
        """
        Bar plot de SHAP importance.
        """

        if model_name not in self.shap_values:
            print(f"⚠️ No hay SHAP calculado para {model_name}.")
            return

        shap.plots.bar(
            self.shap_values[model_name],
            max_display=max_display
        )

    def plot_permutation_importance_histgb(
        self,
        model_name="HistGradientBoosting",
        top_n=25,
        max_samples=5000,
        n_repeats=5,
        scoring="average_precision",
        random_state=42,
        figsize=(10, 8)
    ):
        """
        Calcula permutation importance para HistGradientBoosting.

        Como HistGradientBoostingClassifier no tiene feature_importances_,
        usamos permutation importance sobre test.

        scoring recomendado:
        - "average_precision" para problemas desbalanceados.
        - "roc_auc" si quieres separación general.
        - custom_f2 si quieres importancia orientada a F2.
        """

        from sklearn.inspection import permutation_importance
        from sklearn.metrics import fbeta_score
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        if model_name not in self.pipelines:
            print(f"⚠️ {model_name} no está entrenado.")
            return None

        pipeline = self.pipelines[model_name]

        # Submuestreo para que no tarde demasiado
        n = min(max_samples, len(self.X_test))

        idx = np.random.default_rng(random_state).choice(
            len(self.X_test),
            size=n,
            replace=False
        )

        X_eval = self.X_test.iloc[idx].copy()
        y_eval = np.asarray(self.y_test)[idx]

        # Scoring personalizado F2 si se quiere
        if scoring == "custom_f2":

            threshold = self.thresholds.get(model_name, 0.5)

            def scorer_f2(estimator, X, y):
                proba = estimator.predict_proba(X)[:, 1]
                pred = (proba >= threshold).astype(int)
                return fbeta_score(
                    y,
                    pred,
                    beta=2.0,
                    zero_division=0
                )

            scoring_used = scorer_f2

        else:
            scoring_used = scoring

        print(f"🔎 Calculando permutation importance para {model_name}...")
        print(f"   Muestras usadas: {n}")
        print(f"   Scoring: {scoring}")

        result = permutation_importance(
            pipeline,
            X_eval,
            y_eval,
            scoring=scoring_used,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        importancias = (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        print(importancias.head(top_n))

        plt.figure(figsize=figsize)
        imp_plot = importancias.head(top_n).sort_values("importance_mean")

        plt.barh(
            imp_plot["feature"],
            imp_plot["importance_mean"],
            xerr=imp_plot["importance_std"]
        )

        plt.title(f"Permutation importance - {model_name}")
        plt.xlabel("Caída media de la métrica al permutar la variable")
        plt.tight_layout()
        plt.show()

        return importancias


    def compute_shap_histgb(
        self,
        model_name="HistGradientBoosting",
        max_samples=300,
        background_size=100,
        random_state=42
    ):
        """
        Calcula SHAP para HistGradientBoosting usando una función predict_proba.

        Es más lento que TreeExplainer, pero sirve para explicar el modelo.
        Se usa una muestra pequeña para hacerlo manejable.
        """

        import numpy as np
        import pandas as pd

        if not SHAP_AVAILABLE:
            print("⚠️ SHAP no está disponible. Instala con: pip install shap")
            return None

        if model_name not in self.pipelines:
            print(f"⚠️ {model_name} no está entrenado.")
            return None

        pipeline = self.pipelines[model_name]

        rng = np.random.default_rng(random_state)

        sample_size = min(max_samples, len(self.X_test))
        background_size = min(background_size, len(self.X_train))

        sample_idx = rng.choice(
            len(self.X_test),
            size=sample_size,
            replace=False
        )

        bg_idx = rng.choice(
            len(self.X_train),
            size=background_size,
            replace=False
        )

        X_sample = self.X_test.iloc[sample_idx].copy()
        X_background = self.X_train.iloc[bg_idx].copy()

        def predict_positive(data):
            data_df = pd.DataFrame(
                data,
                columns=self.feature_names
            )
            return pipeline.predict_proba(data_df)[:, 1]

        print(f"🎯 Calculando SHAP para {model_name}...")
        print(f"   Background: {background_size}")
        print(f"   Muestra explicada: {sample_size}")

        explainer = shap.Explainer(
            predict_positive,
            X_background
        )

        shap_values = explainer(X_sample)

        self.shap_values[model_name] = shap_values

        print(f"✅ SHAP completado para {model_name}")

        return shap_values


    def plot_shap_histgb_summary(
        self,
        model_name="HistGradientBoosting",
        max_display=20
    ):
        """
        Beeswarm plot para HistGradientBoosting.
        """

        if model_name not in self.shap_values:
            print(f"⚠️ No hay SHAP calculado para {model_name}. Ejecuta primero compute_shap_histgb().")
            return

        shap.plots.beeswarm(
            self.shap_values[model_name],
            max_display=max_display
        )


    def plot_shap_histgb_bar(
        self,
        model_name="HistGradientBoosting",
        max_display=20
    ):
        """
        Bar plot de importancia media absoluta SHAP.
        """

        if model_name not in self.shap_values:
            print(f"⚠️ No hay SHAP calculado para {model_name}. Ejecuta primero compute_shap_histgb().")
            return

        shap.plots.bar(
            self.shap_values[model_name],
            max_display=max_display
        )

        def compute_shap(self, model_names=None, max_samples=200, background_size=100):
            if not SHAP_AVAILABLE:
                print("⚠️ SHAP no disponible.")
                return

            if model_names is None:
                model_names = list(self.pipelines.keys())

            if isinstance(model_names, str):
                model_names = [model_names]

            sample_size = min(max_samples, self.X_test.shape[0])
            background_size = min(background_size, self.X_train.shape[0])

            sample_idx = np.random.choice(
                self.X_test.shape[0],
                sample_size,
                replace=False
            )

            bg_idx = np.random.choice(
                self.X_train.shape[0],
                background_size,
                replace=False
            )

            X_sample = self.X_test.iloc[sample_idx]
            X_background = self.X_train.iloc[bg_idx]

            print("\n🎯 Calculando SHAP...\n")

            for model_name in model_names:
                if model_name not in self.pipelines:
                    continue

                print(f"▶️ SHAP para {model_name}...")

                try:
                    pipeline = self.pipelines[model_name]

                    # -------------------------------
                    # Redes neuronales tabulares
                    # -------------------------------
                    if model_name == "Neural Network":
                        scaler = pipeline["scaler"]
                        nn_model = pipeline["model"]

                        X_bg_scaled = scaler.transform(X_background)
                        X_sample_scaled = scaler.transform(X_sample)

                        explainer = shap.Explainer(
                            nn_model.predict,
                            X_bg_scaled
                        )

                        shap_values = explainer(X_sample_scaled)

                    # -------------------------------
                    # Árboles
                    # -------------------------------
                    elif model_name in ["Random Forest", "XGBoost"]:
                        model = self.models[model_name]

                        explainer = shap.TreeExplainer(model)
                        shap_values_raw = explainer.shap_values(X_sample)

                        if isinstance(shap_values_raw, list):
                            shap_values_raw = shap_values_raw[1]

                        shap_values = shap.Explanation(
                            values=shap_values_raw,
                            data=X_sample.values,
                            feature_names=self.feature_names
                        )

                    # -------------------------------
                    # Lineales y SVM
                    # -------------------------------
                    else:
                        def predict_positive(data):
                            data_df = pd.DataFrame(
                                data,
                                columns=self.feature_names
                            )
                            return pipeline.predict_proba(data_df)[:, 1]

                        explainer = shap.Explainer(
                            predict_positive,
                            X_background
                        )

                        shap_values = explainer(X_sample)

                    self.shap_values[model_name] = shap_values

                    print(f"   ✅ SHAP completado para {model_name}\n")

                except Exception as e:
                    print(f"   ❌ Fallo SHAP en {model_name}: {e}\n")

    def plot_shap_summary(self, model_name):
        if model_name not in self.shap_values:
            print(f"⚠️ No hay SHAP calculado para {model_name}")
            return

        shap.plots.beeswarm(self.shap_values[model_name], max_display=20)