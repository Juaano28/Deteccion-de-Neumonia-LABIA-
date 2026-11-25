# Resultado del experimento MobileNetV2

- Datos: 3796 entrenamiento, 950 validación, 624 test oficial.
- Selección: mejor AUROC de validación, etapa finetune, época global 15.
- Umbral fijado en validación: 0.87450653; objetivo de sensibilidad: 90%.
- Test oficial: sensibilidad 94.10 %; especificidad 85.90 %;
  precisión 91.75 %; accuracy 91.03 %.
- Errores test oficial: FN=23 y FP=33;
  aciertos TP=367 y TN=201.
- AUROC test oficial: 0.9653; AP: 0.9696;
  Brier: 0.0888; ECE (10 intervalos): 0.0956.
- Vista única del mismo test: 618 imágenes;
  sensibilidad 94.06 %; especificidad 85.71 %;
  AUROC 0.9648. Es una vista relacionada, no validación externa.
- No se ajustaron parámetros ni umbral utilizando estas métricas de test.
- La salida sigmoide es un puntaje no calibrado; no es la probabilidad individual de acertar.
- La independencia clínica de pacientes no está verificada. El alcance es académico.
- Huella de datos: b34c6fd669e98231fdcc8f772503197097b846d36ef9772645f43ce0ba82d8c0.
- Modo de prueba técnica: False.
