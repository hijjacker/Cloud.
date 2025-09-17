# Imports
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import time

# Load Model Function
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model/model_unquant.tflite")
    interpreter.allocate_tensors()
    with open("model/labels.txt", "r") as f:
        class_names = [line.strip()[2:] for line in f.readlines()]
    return interpreter, class_names

# Project Title
st.markdown(
    """
    <div style = 'text-align: center;
                font-size: 250px; 
                font-weight: bold; 
                font-family: Helvetica, sans-serif; 
                color: #1F618D;'>
        Cloud.
    </div>
    """,
    unsafe_allow_html=True
)

tab_cover, tab_explanation, tab_gallery, tab_steinbeck, tab_clouds_form, tab_weather, tab_ocean, tab_classifier = st.tabs([
    "Project Cover", "Project Explanation", "Personal Cloud Image Gallery", 
    "Steinbeck and Clouds", "How Clouds Form", "Clouds and Weather", 
    "Clouds and the Ocean", "Cloud Image Classifier"
])

with tab_cover:
    # SoCo Credit
    st.markdown(
        """
        <div style ='text-align: center; 
        font-size: 40px;
        font-weight: bold;
        font-family: Helvetica, sans-serif;
        color: #7299a1 '>
        SoCo 2025: Discovery Monterey Bay
        </div>
        """,
        unsafe_allow_html=True
    )
    # My name
    st.markdown(
        """
        <h2 style ='text-align: center; '>
        Jack Flynn
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Empty space
    st.write("")

    # Steinbeck Quote
    st.markdown(
        """
        <h3 style='text-align: center; font-style: italic;'>
        “The clouds appeared and went away, and in a while they did not try anymore.”
        </h3>
        <p style='text-align: center; font-size: 16px;'>
        — John Steinbeck, <em>The Grapes of Wrath</em>
        </p>
        """,
        unsafe_allow_html=True
    )

# Main Page
with tab_explanation:
    st.subheader("Why I Chose This Project")
    st.write("""
    Before this SoCo, I spent this past summer in Boulder, CO, working at an educational non-profit. Coming from the Bay, one of the first things I noticed when I arrived in Boulder was the difference in cloud type and size. The clouds in Boulder were taller, wider, and unpredictable. One moment, it could be raining and less than a minute later it would be sunny. 

    While in Boulder, I checked out the National Center for Atmospheric Research, which is headquarted there. The campus itself is very famous: it was designed by I.M. Pei (the same architect who designed the Louvre) to resemble Stonehenge.

    There, I learned that Boulder has these interesting cloud formations because of the interaction between the towering Front Range on its West flank and the five-hundred-mile march of the Great Plains to the East. That sense of scale really piqued my interest in weather.

    My interest kept growing while I was here in Monterey. Our meteorology lecture by David Ortiz was my favorite, especially because everything he talked about could be seen in the window behind him. Learning about marine fog and the land-breeze cycle made it clear to me I would focus on weather for my final project.

    As it turns out, I have a lot of hobbies that rely on the weather. If I'm backcountry skiing, for example, getting a live weather forecast is usually impossible; even if available, they can be inaccurate to the specific area you're in. Or in paragliding, where it’s beyond important to understand concepts of winds aloft, thermals, and ridge lift lest you stall or spiral with no warning. Weather forecasts can’t capture the dynamism of a small area like actual visual information, such as building cumulus clouds and fast-moving cirrus clouds, can.

    Most importantly, though, clouds are magnificent. They can span a vertical distance of over 66,000 feet, or more than twice that of Mt. Everest. Our sky would be utterly boring without clouds. They are intricately detailed and inspire awe. That they can also be used to learn about the weather is just a bonus.
    """
    )

    col1, col2 = st.columns(2)

    # Images
    with col1:
        st.image("NCAR.jpg", "National Center for Atmospheric Research in Boulder, CO")
    
    with col2:
        st.image("Boring.jpg", "BORING!")

    st.subheader("How I Made This Project")
    st.write("""
            I used Streamlit to host and run the project website. Streamlit is a super beginner-friendly way to draw up webapps using just Python scripts, something that I'm more comfortable in. I installed Streamlit, then made a file in VS Code and did everything in there. For the image classifier tool, I used Google Teachable Machine. I'll go into more detail on that page. 
            I've fiddled around with Streamlit once, but never published anything, so this was a new experience for me. I only have minimal experience with coding, but the website itself was a super fun/manageable challenge. The trickiest part was definitely incorporating the GTM model, which became a little more technical than I expected.
             """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("Streamlit_Code.jpg", "Only 385 lines of Python! Pretty powerful stuff.")
    with col2:
        st.image("Streamlit.png")

### Image Gallery Label
with tab_gallery:
    st.markdown(
        """
        <h2 style ='text-align: center; '>
        Personal Image Gallery
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Define three column layout
    col1, col2, col3 = st.columns(3, gap="Small")

    # Image Gallery
    with col1:
        st.image("BoulderAnvil.JPEG", "Cumulonimbus at Sunset\n\n Location: Boulder, CO")
        st.image("CCMB.JPEG", "Cirrocumulus/Cirrostratus at Noon\n\n Location: Monterey Bay, CA")
        st.image("PPAnvil.JPG", "Distant Cumulonimbus in Afternoon\n\n Location: Pikes Peak, CO")
        st.image("JMT_Clouds.JPEG", "Iridescent Cirrus Cloud\n\n Location: Muir Pass, CA")
        st.image("Airbnb_Clouds.JPEG", "Wispy Stratus Clouds at Sunset\n\n Location: Boulder, CO")

    with col2:
        st.image("Kilimanjaro_Clouds.jpg", "Low-lying Stratus in Early Morning\n\n Location: Mt. Kilimanjaro, TZ")
        st.image("GreenPeak_Clouds.JPG", "Building Cumulus in Late Afternoon\n\n Location: Boulder, CO")
        st.image("ShastaCirrus_Clouds.JPG", "Fast-Moving Cirrus Clouds near 14,000 feet\n\n Location: Mt. Shasta, CA")
        st.image("Turret_Clouds.jpg", "Large Cirrus Castellanus Cloud\n\n Location: Monterey Bay, CA")
        st.image("Longs_Peak_Clouds.jpg", "Altocumulus Clouds at Sunrise\n\n Location: Longs Peak, CO")
        st.image("Third_Flatiron.JPG", "Cumulus Mediocris at Sunset\n\n Location: Boulder, CO")
        st.image("ShastaMisery.JPG", "Cirrus Uncinus Clouds\n\n Location: Mt. Shasta, CA")


    with col3:
        st.image("Virgil.JPG", "Virgil falling from Cumulus at Sunset\n\n Location: Salida, CO")
        st.image("MarineLayer.jpg", "Distant Coastal Marine Layer in Late Afternoon\n\n Location: Big Sur, CA")
        st.image("Baker_Clouds.JPEG", "Low-lying Stratus Beneath Stratocumulus Layer \n\n Location: Mt. Shuksan, WA")
        st.image("PikesPeak.JPG", "Developing Cumulus Calvus \n\n Location: Pikes Peak, CO")
        st.image("Pileus.jpg", "Cumulus Clouds with Lenticular Caps (Pileus)\n\n Location: Boulder, CO")
        st.image("Mt.Adams.jpg", "Stratocumulus Clouds in Afternoon\n\n Location: Mt. Adams, WA")
        st.image("Tahoe_Clouds.jpg", "Low-Lying Stratus from Lake Effect\n\n Location: Lake Tahoe, CA")

### What clouds mean
with tab_weather:
    st.markdown(
        """
        <h1 style ='text-align: center; '>
        How Clouds Can Be Useful
        </h1>
        """,
        unsafe_allow_html=True
    )

    # General image
    st.image("CloudClassifier.jpg", "Cheat Sheet For Major Types of Clouds")
    
    # Individual images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("CirrusPredict.jpg", "Cirrus - Fair Weather, Possible Winds")
        st.image("AltocumulusPredict.jpg", "Altocumulus - Precedes Afternoon Thunderstorms")
        st.image("NimbostratusPredict.jpg", "Nimbostratus - Rain/Snow")
    
    with col2:
        st.image("CirrocumulusPredict.jpg", "Cirrocumulus - Fair Weather")
        st.image("Altostratus.jpeg", "Altostratus - Stormy Weather")
        st.image("StratusPredict.jpg", "Stratus - Possible Fog/Drizzle")
    
    with col3:
        st.image("CirrostratusPredict.jpg", "Cirrostratus - Possible Precipitation in 24 Hours")
        st.image("StratocumulusPredict.jpg", "Stratocumulus - Fair Weather")
        st.image("CumulusPredict.jpg", "Cumulus -- Fair weather, Unless...")
    
    st.image("CumulonimbusPredict.jpg", "It Builds Into a Thunderhead!")

### Monterey Bay: Clouds and the Ocean
with tab_ocean:
    st.markdown(
        """
        <h1 style ='text-align: center; '>
        How Does the Ocean Affect Coastal Clouds?
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style ='text-align: center; '>
        Aerosols
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.image("Aerosols.jpg", "Diagram of Cloud-Seeding Ocean Processes")

    st.write("""
The ocean is a major source of aerosols, which are the tiny particles suspended in the atmosphere that clouds form around. When waves break and bubbles burst at the sea surface, they release microscopic droplets containing salts, organic compounds, and other materials. These particles are lifted into the air, where they act as cloud condensation nuclei, giving water vapor a surface to condense onto. Without these aerosols, clouds wouldn't be able to form, and precipitation patterns could be drastically different.

The ocean emits organic and inorganic aeosols. The type of aerosols produced by the ocean at a given place depends on biological activity and physical conditions at the sea surface. In biologically active zones near the coast, phytoplankton release sulfuric compounds called DMS that can become aerosols. In stormy areas, strong winds increase bubble formation and spray, which act as aerosols. Ocean-derived aerosols not only influence cloud properties like brightness, they also affect global climate by reflecting up to 30 percent of sunlight and regulating the Earth’s temperature.

             """
             )

    st.markdown(
        """
        <h3 style ='text-align: center; '>
        Land Breeze Cycle
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.image("LandBreeze.jpg", "Diagram of Land-Breeze Cycle")

    st.write("""The land breeze cycle is one of the most important process for coastal cloud formation. It is simple to understand. During the daytime, air over land warms faster than air from the ocean. That air begins to rise over the land, and forms a cloud. Cold air from the ocean rushes in to fill its place, while the hot air above heads in the direction of the ocean, where it cools and falls back to sea level. It is then heated by the sun and begins to move back to land. Clouds thus form over land and move toward the ocean.
             This process is reversed during nighttime, when the ocean is warmer than the land. In this case, clouds form over the ocean and move toward the land.
             """)

    st.markdown(
        """
        <h3 style ='text-align: center; '>
        Temperature and Moisture
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.image("Temperature:Moisture.jpg", "Diagram of Temperature Effect on Thunderheads")

    st.write("""
    In the tropics, temperature and humidity can drive huge cloud formation. Warm air near the surface heats rapidly under a more intense sun. This air becomes more buoyant and rises further into the atmosphere than in colder climates. Because warm air can hold more water vapor, tropical air masses often carry large amounts of invisible moisture. As this air ascends, it cools and reaches its dew point, causing water vapor to condense into tiny droplets - clouds around aerosols that are also often produced by the ocean. This process is especially powerful in the tropics, where high temperature and abundant evaporation keep feeding the atmosphere with heat and moisture.

    Humidity then determines how thick and tall these clouds become. In very humid conditions, rising parcels of air condense quickly and release latent heat, which then can fuel stronger upward motion. This creates towering cumulonimbus clouds that dominate tropical skies and drive thunderstorms and heavy rain. Conversely, if the air aloft is dry, clouds can evaporate as quickly as they form, leading to thinner and less stable formations.

    """)
    st.markdown(
        """
        <h3 style ='text-align: center; '>
        Monterey Bay and California
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.image("MarineLayer.png", "Effects of Coastal Marine Layer on Temperature")

    st.write("""
             Monterey Bay's coastal clouds become very obvious after spending a few days in the area. This weather phenomenon is called the coastal marine layer. It is shaped by the land-breeze cycle. The ocean warms and cools more slowly than the land, so air just above the water often holds more moisture and stays relatively stable in temperature. When this moist marine air meets warmer air from the land, it cools and condenses into clouds.

             In Monterey Bay, this interaction produces the region's constant fog and low marine clouds. Cooler ocean waters, fed by upwelling from the California Current, chill the air above and increase relative humidity. When that saturated air is pushed inland by winds, it condenses into dense coastal stratus that can blanket the shoreline for much of the day.
             """
             )

### Image Classifier
with tab_classifier:
    st.image("GTM.png", "Big Dawg")
    st.markdown(
        """
        <h1 style ='text-align: center; '>
        Cloud Classifier Using Google Teachable Machine
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Get Image File
    img_file = st.camera_input("Take a picture of a cloud!")

    if img_file:
        # Fake loading for vanity haha
        with st.spinner("Model loading", show_time=True):
            model, class_names = load_tflite_model()
            time.sleep(3)

        # Open and preprocess
        image = Image.open(img_file).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

        # Predict
        interpreter, class_names = load_tflite_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        index = np.argmax(output_data)
        confidence = output_data[0][index]

        # Show results
        st.header(f"**Prediction:** {class_names[index]}")
        st.subheader(f"**Confidence:** {confidence*100:.2f}%")

    st.header("How I Made This")
    st.write("I used Google Teachable Machine to train this model, which is pretty much the easiest way to go about it. It only required images to actually train the model (exporting it was a little trickier!). I trained it on five different classes, cirrus, cirrocumulus, cumulus, cumulonimbus, and stratus. I uploaded ~50 images of each, a small number of which were my own, but most came from the web. After testing it and adjusting a few parameters, I tried to export it. Usually, to use it in a webapp you just copy paste the javascript that GTM provides. However, I made this website using Streamlit, so I had to figure out how to make it compatabile with my python script. I first tried exporting it using a Keras file. I tried to incorporate this into my code, but ran into threading issues, as Python3.13 isn't compatabile with tensorflow. This took me a long time to figure out, because I'm not a technical person. I ended up using a TensorflowLite file which only runs inference on the model from the cloud (doesn't actually host it), which is slightly less accurate but solved the threading issue. And to my surprise it worked! Definitely the most difficult part of the project, but the most worthwhile!")
    st.image("Model.jpg", "Training the GTM Model")


### Steinbeck Quotes Page
with tab_steinbeck:
    st.markdown(
        """
        <h1 style ='text-align: center; '>
        Steinbeck and Clouds
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.write("Clouds had a noticeable impact on many of Steinbeck's writings. For his novels about agriculture, they were a double-edged sword. They served as both a source of abundance, bringing rain to thirsty crops, and a source of destruction, flooding towns or not raining at all. Below are some quotes from Steinbeck's novels that encapsulate the influence clouds had on his work.")    
    st.image("steinbeck_image.jpg", "John Steinbeck")

    col1,col2 = st.columns(2)

    with col1:
        st.write("“A large drop of sun lingered on the horizon and then dripped over and was gone, and the sky was brilliant over the spot where it had gone, and a torn cloud, like a bloody rag, hung over the spot of its going.”")
        st.image("Grapes_Wrath.jpg", "Grapes of Wrath, 1939")
        st.write("“It's a cloud,” she said. “There's word snow is on the way, and it's early, too.” Doctor Winter went to the window and squinted up at the sky, and he said, “Yes, it's a big cloud; maybe it will pass over.”")
        st.image("Moon_Down.jpg", "The Moon is Down, 1942")
        st.write("“A cloud drifting in formed the letters O-N in the sky over Monterey.”")
        st.image("Sweet_Thursday.jpg", "Sweet Thursday, 1954")

    with col2:
        st.write("“A blanket of herring clouds was rolling in from the East.”")
        st.image("East_Eden.jpg", "East of Eden, 1952")
        st.write("“There might be clouds in the morning,” Joseph said. “It's so close to the new year, there might be clouds.”")
        st.image("God_Unknown.jpg", "To a God Unknown, 1933")
        st.write("“The clouds were tattering, and there were splashes of lovely clear sky with silks of cloud skittering across them.”")
        st.image("Wayward_Bus.jpg", "The Wayward Bus, 1947")

### How Clouds Form
with tab_clouds_form:
    st.markdown(
        """
        <h1 style ='text-align: center; '>
        How do Clouds Form?
        </h1>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dew Point", "Cloud Condensation Nuclei", "Adiabatic Cooling", "Convection", "Frontal Lifting", "Orographic Lifting"])

    with tab1:
        st.header("Dew Point")
        st.image("DewPoint.jpeg", "Diagram of Water Vapor Reaching its Dew Point")
        st.write("Clouds form when rising air cools to its dew point, the temperature where the air becomes saturated and can no longer hold all of its water vapor. As warm air rises into the atmosphere, it expands because of lower pressure at higher altitudes, which makes it cool down. Once the air cools to the dew point, water vapor condenses into countless tiny liquid water droplets (or ice crystals, if it's cold enough). These droplets are what we see as a cloud. Air rises, cools, reaches saturation, and condensation creates a visible cloud.")
    
    with tab2:
        st.header("Cloud Condensation Nuclei")
        st.image("CCN.jpg", "Diagram of Cloud Condensation Nuclei")
        st.write("For condensation to actually happen, though, clouds need cloud condensation nuclei (CCN). CCN are microscopic particles in the atmosphere, such as dust, sea salt, smoke, or even pollution, that provide surfaces for water vapor to cling to. Water vapor alone usually won't condense into droplets, but with CCN present, the molecules can form stable droplets. The more CCN in the air, the more droplets can form, which influences cloud thickness, brightness, and whether it rains.")

    with tab3:
        st.header("Adiabatic Cooling")
        st.image("Adiabatic.jpg", "Diagram of Adiabatic Cooling")
        st.write("A major driver of cloud development is adiabatic cooling, which happens when air rises and expands without exchanging heat with its surroundings. As air moves upward, the pressure around it decreases, so it expands and cools. Cooler air can hold less water vapor, so once it cools enough to reach saturation, condensation begins and clouds form. This process is especially important in mountainous regions, where air is forced to rise over terrain, and in weather systems where air is lifted by fronts or convection.")

    with tab4:
        st.header("Convection")
        st.image("Convection.jpg", "Diagram of Convection in Cloud Formation")
        st.write("Clouds often form through convection, when the sun heats the Earth's surface unevenly. Warmer patches of ground heat the air above them, making it rise in “bubbles” of warm air called thermals. As these thermals rise, they cool adiabatically, and if they reach the dew point, condensation occurs, producing cumulus clouds. This process explains why puffy, fair-weather clouds often appear during the afternoon when the ground has warmed the most.")

    with tab5:
        st.header("Frontal Lifting")
        st.image("Frontal.jpg", "Diagram of Frontal Lifting")
        st.write("Another important mechanism is frontal lifting, which occurs when two air masses with different temperatures meet. Warm, less dense air is forced upward over cold, dense air along a front. As the warm air rises, it cools and can reach saturation, forming clouds. This is why large sheets of clouds often form along cold fronts and warm fronts, leading to storms.")

    with tab6:
        st.header("Orographic Lifting")
        st.image("Orographic.jpg", "Diagram of orographic lifting in cloud formation", width = 500)
        st.write("Cloud development is also influenced by orographic lifting, which happens when moving air encounters a mountain or hill and is forced to rise. As the air ascends the slope, it cools adiabatically and can reach the dew point, producing clouds along the mountain’s windward side. On the leeward side, the air descends, warms, and dries, creating a “rain shadow” region with little cloud cover.")
