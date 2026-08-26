# Solution for #254: Client fails to fetch prompts with names containing slash

a/promptlayer/utils.py
+++ b/promptlayer/utils.py
@@ -915,7 +915,8 @@
     if not prompt_name:
         raise ValueError("prompt_name is required")
-    url = f"{BASE_URL}/prompts/{prompt_name}"
+    from urllib.parse import quote
+    url = f"{BASE_URL}/prompts/{quote(prompt_name, safe='')}"
     params = {}
     if version is not None:
         params["version"] = version
