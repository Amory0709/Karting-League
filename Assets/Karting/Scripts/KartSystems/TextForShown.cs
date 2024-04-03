using KartGame.KartSystems;
using TMPro;
using UnityEngine;

public class TextForShown : MonoBehaviour
{
    public TextMeshPro PlayerNameShown { get; private set; }
    public ArcadeKart Kart { get; private set; }
    private Transform viewCamera;

    void Awake()
    {
        Kart = GetComponentInParent<ArcadeKart>();
        PlayerNameShown = GetComponent<TextMeshPro>();
        PlayerNameShown.text = Kart?.PlayerName;
    }

    // Start is called before the first frame update
    void Start()
    {
        // Find the view camera by searching for a GameObject named "Main camera"
        GameObject cameraObject = GameObject.Find("Main Camera");

        // Check if the camera object was found
        if (cameraObject != null)
        {
            // Get the camera's transform component
            viewCamera = cameraObject.transform;
        }
        else
        {
            Debug.LogError("Main camera not found!");
        }

        PlayerNameShown.rectTransform.sizeDelta = new Vector2(20, PlayerNameShown.rectTransform.sizeDelta.y);
        PlayerNameShown.fontSize = 28;
    }

    // Update is called once per frame
    void Update()
    {
        if (viewCamera != null)
        {
            // Calculate the desired rotation based on the view camera's rotation
            //Quaternion desiredRotation = Quaternion.LookRotation(PlayerNameShown.transform.position - viewCamera.position);
            Quaternion desiredRotation = Quaternion.Euler(viewCamera.eulerAngles.x, viewCamera.eulerAngles.y, 0f);

            // Apply the rotation to the Text Mesh Pro object
            PlayerNameShown.transform.rotation = desiredRotation;
        }
    }
}
