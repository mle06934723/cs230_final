import random 
from data import BinaryRewardModelDataset 
from torch.data.utils import DataLoader 
from clean_labels import compute_features, pair_selection 

positives = [
  "Art is the most beautiful way to express emotions that cannot be conveyed through words.",
  "The beauty of a painting lies not just in its colors, but in the story it tells.",
  "A well-curated piece of artwork can transform a room, elevating its aesthetic appeal.",
  "Beauty is often found in the smallest details, like the delicate brushstrokes in a watercolor painting.",
  "The interplay of light and shadow can create dramatic beauty in a photograph.",
  "A sculpture's form and texture evoke a sense of movement and life in a static medium.",
  "Nature is often the most profound source of inspiration for artists, with its vivid colors and intricate patterns.",
  "A great piece of art captures the essence of time, place, and culture in a way that transcends words.",
  "The beauty of minimalist design lies in its simplicity, where every element has purpose.",
  "There’s something timeless about the way classical architecture blends form with function.",
  "The subtle curves in a classical marble statue reveal the artist’s mastery over human anatomy.",
  "Art can be a window into a different world, giving viewers a glimpse of another's perspective.",
  "Abstract art speaks to the soul through color and texture, bypassing the need for clear representation.",
  "The contrast between light and dark in a painting can symbolize the tension between opposing forces.",
  "The appreciation of beauty can be an intensely personal experience, depending on cultural and emotional context.",
  "The aesthetic value of a landscape painting lies in its ability to evoke feelings of tranquility and wonder.",
  "The intricacy of fine jewelry designs can be viewed as small-scale sculptures that enhance personal beauty.",
  "The beauty of symmetry in design often resonates deeply with human sense of order and balance.",
  "Graffiti, once considered a form of vandalism, is now recognized as an important expression of urban aesthetics.",
  "Photography allows us to freeze moments of beauty, capturing fleeting emotions in a single frame.",
  "A great work of art does not just reflect beauty, but also challenges our perceptions of the world.",
  "The timeless beauty of a classic portrait comes from its ability to reveal the subject's personality and spirit.",
  "The use of vibrant color in a painting can completely transform the emotional tone of the artwork.",
  "An artist’s use of space and perspective can make the viewer feel both the vastness and intimacy of the scene.",
  "A masterpiece is not just a beautiful object but an intellectual experience that sparks thought and conversation.",
  "The balance of light, texture, and color in an impressionist painting creates a mood that can feel almost tangible.",
  "Beauty is not just something seen; it is something felt, experienced through all the senses.",
  "A well-composed photograph can reveal the hidden beauty of everyday life.",
  "The symmetry of the human face is often regarded as the epitome of physical beauty.",
  "Art is a conversation between the creator and the viewer, where each brings their own interpretation.",
  "The way an artist manipulates medium and material can dramatically alter the aesthetic impact of their work.",
  "A beautiful piece of music has the power to transport the listener to another time and place.",
  "Fashion design is a form of wearable art, where creativity and aesthetics merge to produce beauty.",
  "A graceful dance performance is an aesthetic expression of human movement and emotion.",
  "The beauty of a sunset is often captured in art to evoke feelings of peace and reflection.",
  "The process of creating art is as much about exploration and discovery as it is about achieving beauty.",
  "Beauty is not always something perfect, but something that resonates emotionally.",
  "The combination of sharp lines and smooth curves can create a visual harmony in sculpture.",
  "The allure of a vintage painting often lies in its age, history, and the story it carries.",
  "The aesthetics of nature are reflected in the delicate petals of a flower or the power of a crashing wave.",
  "Visual art serves as a means of communication that goes beyond language, transcending cultural barriers.",
  "The mood of a painting can change dramatically depending on the artist's choice of colors and brushwork.",
  "Fashion is a reflection of the beauty ideals of a given time, culture, and place.",
  "The juxtaposition of old and new in architectural design can create visually dynamic spaces.",
  "The emotional power of a well-composed poem can be considered a form of literary beauty.",
  "The way artists play with light in a painting can give depth and texture to a flat canvas.",
  "A beautiful sculpture does not only engage the eyes, but also invites touch and interaction.",
  "The graceful curves of classical art often embody harmony and balance.",
  "Art provides an outlet for the artist’s personal interpretation of beauty and meaning.",
  "The beauty of art lies in its ability to make the intangible feel real, creating a connection between artist and audience.",
]

negatives = [
    "Dental hygiene involves the practice of maintaining clean teeth and gums to prevent oral diseases, including regular brushing, flossing, and professional cleanings.",
  "Periodontitis is a severe gum infection caused by the accumulation of plaque and tartar, which can lead to gum recession, tooth loss, and bone deterioration.",
  "Root canal therapy is a procedure used to treat infection or damage within the pulp of a tooth, where the infected tissue is removed, and the space is sealed to prevent further infection.",
  "Dental crowns are caps placed over a damaged or decayed tooth to restore its shape, strength, and appearance, typically made from porcelain, metal, or a combination of both.",
  "The dental implant procedure involves the insertion of a titanium post into the jawbone to replace a missing tooth, followed by the attachment of an artificial crown.",
  "Amalgam fillings are a type of dental filling made from a mixture of silver, mercury, tin, and copper, often used to fill cavities in posterior teeth due to their durability.",
  "Composite resin fillings are tooth-colored materials used for restoring decayed or damaged teeth, often preferred for aesthetic purposes due to their ability to blend with natural tooth color.",
  "Dental veneers are thin layers of porcelain or resin placed on the front surface of teeth to improve their appearance, commonly used for discoloration, chips, or gaps.",
  "Tooth extraction is the removal of a tooth from its socket in the bone, typically due to severe decay, infection, or orthodontic treatment.",
  "Braces are orthodontic appliances consisting of brackets, wires, and rubber bands that gradually move teeth into their correct positions, improving alignment and bite.",
  "The bitewing X-ray is a diagnostic tool used to examine the upper and lower teeth in a specific area of the mouth, providing information about decay between teeth and bone level.",
  "Full-mouth X-rays (panoramic radiographs) offer a comprehensive view of the teeth, jaws, sinuses, and surrounding structures to help detect issues like impacted teeth, tumors, and bone diseases.",
  "Teeth whitening is a cosmetic procedure that involves the application of bleaching agents such as hydrogen peroxide or carbamide peroxide to remove stains and lighten the color of teeth.",
  "Dental sealants are thin, plastic coatings applied to the chewing surfaces of molars to prevent cavities by sealing out food particles and bacteria.",
  "Gingival grafting is a periodontal procedure where tissue is taken from the roof of the mouth or from a donor site and grafted onto the gums to treat gum recession.",
  "TMJ (temporomandibular joint) disorders involve dysfunction of the jaw joint, which can lead to pain, headaches, and clicking sounds during mouth movement.",
  "Dental bridges are prosthetic devices used to replace one or more missing teeth by attaching artificial teeth to adjacent healthy teeth or dental implants.",
  "Invisalign is a brand of clear, removable aligners used in orthodontics to straighten teeth gradually, offering an alternative to traditional metal braces.",
  "Crown lengthening is a surgical procedure in which excess gum tissue or bone is removed to expose more of the tooth, often performed before placing a crown or filling.",
  "Fluoride treatments help remineralize enamel, strengthen teeth, and reduce the risk of cavities, typically administered as a gel, foam, or varnish during dental visits.",
  "Orthodontic spacers are small rubber or metal devices used to create space between teeth before the installation of braces or other orthodontic appliances.",
  "Periodontal scaling and root planing is a non-surgical procedure used to treat gum disease, where plaque, tartar, and bacteria are removed from below the gumline to promote healing.",
  "Dental resin-bonded bridges use a combination of composite resin and metal wings to bond artificial teeth to adjacent teeth, providing a conservative solution to tooth loss.",
  "Zirconia crowns are a type of dental crown made from zirconium oxide, known for their strength, durability, and aesthetic appeal, often used for back teeth restorations.",
  "Cone beam computed tomography (CBCT) provides three-dimensional imaging for dental professionals to plan procedures like dental implants, extractions, and jaw surgeries with high precision.",
  "Laser dentistry utilizes focused laser light to perform various dental procedures, including soft tissue surgery, cavity preparation, and teeth whitening, with minimal discomfort and recovery time.",
  "Ozone therapy in dentistry is an alternative treatment that uses ozone gas to disinfect and promote healing in cavities, periodontal infections, and soft tissue lesions.",
  "Retainers are custom-made dental appliances worn after orthodontic treatment to maintain the corrected position of the teeth and prevent relapse.",
  "Dental prophylaxis is a preventive cleaning procedure involving the removal of plaque, tartar, and stains from the teeth to maintain oral health and prevent gingivitis.",
  "Oral cancer screening involves examining the mouth for abnormal lesions, lumps, or discoloration, helping detect early signs of cancer in the lips, tongue, cheeks, and throat.",
  "Pulpotomy is a procedure performed on a primary (baby) tooth in children where the infected pulp is removed to save the tooth and prevent further infection.",
  "Onlays are partial crowns that cover a larger portion of a tooth’s surface, typically used when a cavity or fracture extends beyond the cusps but doesn't require a full crown.",
  "Sinus lift surgery is a procedure used to add bone to the upper jaw in preparation for dental implants, often necessary when the sinus cavity is too close to the jawbone.",
  "Cleft palate repair is a surgical procedure performed to correct a congenital gap in the upper lip and palate, often requiring multidisciplinary care, including orthodontic treatment.",
  "Dental occlusion refers to the way the upper and lower teeth come together when the mouth is closed, and issues like malocclusion can lead to pain and difficulty chewing.",
  "Mouthguards are protective devices used to prevent damage to teeth during sports or to alleviate issues like teeth grinding (bruxism) during sleep.",
  "Prosthodontics is a specialized field of dentistry focusing on restoring missing or damaged teeth with crowns, bridges, dentures, or dental implants.",
  "Dental abscesses are localized collections of pus caused by bacterial infections in the teeth or gums, requiring drainage and sometimes root canal therapy or extraction.",
  "Bleeding gums can be a sign of gingivitis or more severe periodontal disease, indicating the need for professional cleaning and improved at-home oral hygiene practices.",
  "Oral thrush is a fungal infection caused by an overgrowth of Candida albicans, which can affect the mouth and cause painful lesions on the tongue and inner cheeks.",
  "Bone grafting in dentistry involves the placement of bone material to regenerate bone loss in the jaw, often done before implant placement in cases of insufficient bone density.",
  "Tooth sensitivity is a common condition caused by the exposure of dentin or enamel wear, leading to discomfort from hot, cold, or sweet stimuli, often treated with desensitizing agents.",
  "Dental fluorosis occurs due to excessive fluoride exposure during tooth development, leading to discoloration and defects in the enamel's structure.",
  "Pediatric dentistry focuses on the oral care of children, addressing issues like teething, cavity prevention, and early orthodontic assessments.",
  "Oral sedation is a method of calming patients before dental procedures, often using medications like benzodiazepines, to reduce anxiety and discomfort during treatment.",
  "Temporomandibular joint (TMJ) therapy involves various treatments, including splints, physical therapy, and medications, to alleviate pain and dysfunction in the jaw joint.",
  "Dental photography is used to document cases, track treatment progress, and create visual records for patient education or insurance claims.",
  "Skeletal anchorage systems in orthodontics involve the placement of mini-implants to help move teeth in cases where traditional braces cannot provide enough force.",
  "Custom-fit dentures are removable prosthetic devices made to replace missing teeth and gums, custom designed for each patient’s anatomy for a comfortable fit and natural appearance.",
  "Gum contouring is a cosmetic procedure that reshapes the gum line to improve the appearance of the smile, especially in cases of excessive or uneven gum tissue.",
]

def swap_noisy(positives, negatives, num_elements_to_swap): 
  positive_copy = positives[:]
  negative_copy = negatives[:]
  if len(positive_copy) < num_elements_to_swap or len(negative_copy) < num_elements_to_swap:
      raise ValueError("Both lists must have at least the number of swappable elements")
  positive_indices = random.sample(range(len(positive_copy)), num_elements_to_swap)
  negative_indices = random.sample(range(len(negative_copy)), num_elements_to_swap)
  incorrect_texts = []
  for p_idx, n_idx in zip(positive_indices, negative_indices):
      positive_copy[p_idx], negative_copy[n_idx] = negative_copy[n_idx], positive_copy[p_idx]

  for idx in positive_indices:
    incorrect_texts.append(positive_copy[idx])

  for idx in negative_indices:
    incorrect_texts.append(negative_copy[idx])
  return positive_copy, negative_copy, incorrect_texts 

def assert_overlap(examples, incorrect_texts, dataset):
  confident_examples = (examples != 0).nonzero()
  zero_indices = (examples == 0).nonzero()
  zero_indices = zero_indices.squeeze().tolist()
  classified_texts = []
  for idx in zero_indices:
    classified_texts.append(dataset.texts[idx])
  only_misclassified = list(set(classified_texts) - set(incorrect_texts))
  print("Elements in classified but not in ground truth:", only_misclassified)
  only_true = list(set(incorrect_texts) - set(classified_texts))
  print("Elements in ground truth but not in classified:", only_true)
  overlap = list(set(incorrect_texts) & set(classified_texts))
  # print("Overlap (set intersection):", overlap)
  assert len(overlap) == len(incorrect_texts)
  return overlap

def compute_features(model, temploader, batch_size=1):
    all_feats = torch.rand(len(temploader.dataset), 768).t() # [dim, n]
    with torch.no_grad():
        for batch_idx, (text, _, _) in tqdm(enumerate(temploader), total=len(temploader)):
            features = model.encode(text, convert_to_tensor=True) #[1, dim]
            start_index = batch_idx * batch_size
            end_index = batch_idx * batch_size + batch_size
            all_feats[:, start_index:end_index] = features.data.t()
    return all_feats

noisy_positives, noisy_negatives, incorrect_texts = swap_noisy(positives, negatives, 10)
noisy_dataset = BinaryRewardModelDataset(noisy_positives, noisy_negatives)
noisy_dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False) 

model = SentenceTransformer("mleshen22/hateBERT-cl-rlhf")
model.eval() 
features = compute_features(model, noisy_dataloader) 
examples, pairs = pair_selection(model, features, noisy_dataloaders) 
overlap = assert_overlap(examples, incorrect_texts, dummy_dataset) 



