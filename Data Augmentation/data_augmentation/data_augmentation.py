"""
TODO: docstring
"""
import matplotlib.pyplot as pyplot
import tensorflow
import tensorflow_datasets

class DataAugmentationA:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        (train_ds, val_ds, test_ds), metadata = tensorflow_datasets.load(
            'tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True, as_supervised=True)
        num_classes = metadata.features['label'].num_classes
        print(num_classes)
        get_label_name = metadata.features['label'].int2str
        image, label = next(iter(train_ds))
        _ = pyplot.imshow(image)
        _ = pyplot.title(get_label_name(label))
        IMG_SIZE = 180
        self.resize_and_rescale = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
            tensorflow.keras.layers.Rescaling(1.0/255)])
        result = self.resize_and_rescale(image)
        _ = pyplot.imshow(result)
        print("Min and max pixel values:", result.numpy().min(), result.numpy().max())
        self.data_augmentation = tensorflow.keras.Sequential([
            tensorflow.keras.layers.RandomFlip("horizontal_and_vertical"),
            tensorflow.keras.layers.RandomRotation(0.2)])
        # add the image to a batch
        image = tensorflow.cast(tensorflow.expand_dims(image, 0), tensorflow.float32)
        pyplot.figure(figsize=(10, 10))
        for i in range(9):
            augmented_image = self.data_augmentation(image)
            ax = pyplot.subplot(3, 3, i + 1)
            pyplot.imshow(augmented_image[0])
            pyplot.axis("off")
        aug_ds = train_ds.map(
            lambda x, y: (self.resize_and_rescale(x, training=True), y))
        self.batch_size = 32
        self.AUTOTUNE = tensorflow.data.AUTOTUNE
        train_ds = self.prepare(train_ds, shuffle=True, augment=True)
        val_ds = self.prepare(val_ds)
        test_ds = self.prepare(test_ds)
        model = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tensorflow.keras.layers.MaxPooling2D(),
            tensorflow.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tensorflow.keras.layers.MaxPooling2D(),
            tensorflow.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tensorflow.keras.layers.MaxPooling2D(),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(128, activation='relu'),
            tensorflow.keras.layers.Dense(num_classes)])
        model.compile(
            optimizer='adam',
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        epochs = 5
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        loss, acc = model.evaluate(test_ds)
        print("Accuracy", acc)
        # custom data augmentation
        random_invert = self.random_invert()
        pyplot.figure(figsize=(10, 10))
        for i in range(9):
          augmented_image = random_invert(image)
          ax = pyplot.subplot(3, 3, i + 1)
          pyplot.imshow(augmented_image[0].numpy().astype("uint8"))
          pyplot.axis("off")
        _ = pyplot.imshow(RandomInvert()(image)[0])

    def prepare(self, ds, shuffle=False, augment=False):
        """
        TODO: docstring
        """
        ds = ds.map(
            lambda x, y:
            (self.resize_and_rescale(x), y),
            num_parallel_calls=self.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(1000)
        # batch all datasets
        ds = ds.batch(self.batch_size)
        # use data augmentation only on the training set
        if augment:
            ds = ds.map(
                lambda x, y:
                (self.data_augmentation(x, training=True), y), 
                num_parallel_calls=self.AUTOTUNE)
        # use buffered prefetching on all datasets
        return ds.prefetch(buffer_size=self.AUTOTUNE)

    def random_invert(self, factor=0.5):
        """
        TODO: docstring
        """
        return tensorflow.keras.layers.Lambda(lambda x: self.random_invert_img(x, factor))

    def random_invert_img(self, x, p=0.5):
        """
        TODO: docstring
        """
        if  tensorflow.random.uniform([]) < p:
            x = (255-x)
        else:
            x
        return x

class DataAugmentationB:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        (train_ds, val_ds, test_ds), metadata = tensorflow_datasets.load(
            'tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True, as_supervised=True)
        get_label_name = metadata.features['label'].int2str
        image, label = next(iter(train_ds))
        _ = pyplot.imshow(image)
        _ = pyplot.title(get_label_name(label))
        flipped = tensorflow.image.flip_left_right(image)
        self.visualize(image, flipped)
        grayscaled = tensorflow.image.rgb_to_grayscale(image)
        self.visualize(image, tensorflow.squeeze(grayscaled))
        _ = pyplot.colorbar()
        saturated = tensorflow.image.adjust_saturation(image, 3)
        self.visualize(image, saturated)
        bright = tensorflow.image.adjust_brightness(image, 0.4)
        self.visualize(image, bright)
        cropped = tensorflow.image.central_crop(image, central_fraction=0.5)
        self.visualize(image, cropped)
        rotated = tensorflow.image.rot90(image)
        self.visualize(image, rotated)
        for i in range(3):
          seed = (i, 0)  # tuple of size (2,)
          stateless_random_brightness = tensorflow.image.stateless_random_brightness(
              image, max_delta=0.95, seed=seed)
          self.visualize(image, stateless_random_brightness)
        for i in range(3):
          seed = (i, 0)  # tuple of size (2,)
          stateless_random_contrast = tensorflow.image.stateless_random_contrast(
              image, lower=0.1, upper=0.9, seed=seed)
          self.visualize(image, stateless_random_contrast)
        for i in range(3):
          seed = (i, 0)  # tuple of size (2,)
          stateless_random_crop = tensorflow.image.stateless_random_crop(
              image, size=[210, 300, 3], seed=seed)
          self.visualize(image, stateless_random_crop)
        (train_datasets, val_ds, test_ds), metadata = tensorflow_datasets.load(
            'tf_flowers', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True, as_supervised=True)

    def augment(self, image_label, seed):
        """
        TODO: docstring
        """
        image, label = image_label
        image, label = self.resize_and_rescale(image, label)
        image = tensorflow.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
        # make a new seed
        new_seed = tensorflow.random.experimental.stateless_split(seed, num=1)[0, :]
        # random crop back to the original size
        image = tensorflow.image.stateless_random_crop(
            image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
        # random brightness
        image = tensorflow.image.stateless_random_brightness(
            image, max_delta=0.5, seed=new_seed)
        image = tensorflow.clip_by_value(image, 0, 1)
        return image, label

    def resize_and_rescale(self, image, label):
        """
        TODO: docstring
        """
        image = tensorflow.cast(image, tensorflow.float32)
        image = tensorflow.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = (image / 255.0)
        return image, label

    def visualize(self, original, augmented):
        """
        TODO: docstring
        """
        fig = pyplot.figure()
        pyplot.subplot(1,2,1)
        pyplot.title('Original image')
        pyplot.imshow(original)
        pyplot.subplot(1,2,2)
        pyplot.title('Augmented image')
        pyplot.imshow(augmented)

class DataAugmentationC:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        (train_datasets, val_ds, test_ds), metadata = tfds.load(
            'tf_flowers',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True, as_supervised=True)
        # Create a generator.
        rng = tf.random.Generator.from_seed(123, alg='philox')
        train_ds = (
            train_datasets
            .shuffle(1000)
            .map(f, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))
        val_ds = (
            val_ds
            .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))
        test_ds = (
            test_ds
            .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
            .batch(batch_size)
            .prefetch(AUTOTUNE))

    def augment(image_label, seed):
        image, label = image_label
        image, label = resize_and_rescale(image, label)
        image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
        # Make a new seed.
        new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        # Random crop back to the original size.
        image = tf.image.stateless_random_crop(
            image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
        # Random brightness.
        image = tf.image.stateless_random_brightness(
            image, max_delta=0.5, seed=new_seed)
        image = tf.clip_by_value(image, 0, 1)
        return image, label

    def resize_and_rescale(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = (image / 255.0)
        return image, label

    def f(x, y):
        """
        Wrapper Function for updating seeds.
        """
        seed = rng.make_seeds(2)[0]
        image, label = augment((x, y), seed)
        return image, label

class RandomInvert(tensorflow.keras.layers.Layer):
    """
    TODO: docstring
    """
    def __init__(self, factor=0.5, **kwargs):
        """
        TODO: docstring
        """
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        """
        TODO: docstring
        """
        return DataAugmentationA().random_invert_img(x)

if __name__ == '__main__':
    DataAugmentationC()
