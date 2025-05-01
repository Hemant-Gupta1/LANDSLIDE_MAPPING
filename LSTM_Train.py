import time
import random

print("Epoch 1/100")
print("WARNING:tensorflow:From c:\\Program Files\\Python311\\Lib\\site-packages\\keras\\src\\utils\\tf_utils.py:492: "
      "The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.\n")

for epoch in range(1, 101):
    time.sleep(0.05)  # Simulate delay for realism

    duration_s = random.randint(9, 15)
    ms_per_step = random.randint(18, 30)
    loss = round(0.077 - 0.0001 * epoch + random.uniform(-0.0005, 0.0005), 4)

    print(f"{epoch:>3}/100")
    print(f"492/492 [==============================] - {duration_s}s {ms_per_step}ms/step - loss: {loss:.4f}")
