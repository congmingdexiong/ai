import json

input = "[{'category': 'Smartphones and Accessories', 'products': ['SmartX ProPhone']}, {'category': 'Cameras and Camcorders', 'products': ['FotoSnap DSLR Camera']}, {'category': 'Televisions and Home Theater Systems', 'products': ['CineView 4K TV', 'CineView 8K TV', 'CineView OLED TV']}] \n\nBased on the customer service query, the output is a list of objects. The first object has the category 'Smartphones and Accessories' and the product 'SmartX ProPhone'. The second object has the category 'Cameras and Camcorders' and the product 'FotoSnap DSLR Camera'. The third object has the category 'Televisions and Home Theater Systems' and the products 'CineView 4K TV', 'CineView 8K TV', and 'CineView OLED TV'.\n\nNote that the customer service query did not mention any specific TV model, so all three TV models in the 'Televisions and Home Theater Systems' category are included in the output."
input_string = input.replace("'", "\"").split("\n\n")[0]
# print("\'" + input_string + "\'")
try:
    data = json.loads(input_string)
    print(data)
except json.JSONDecodeError:
        print("Error: Invalid JSON string")